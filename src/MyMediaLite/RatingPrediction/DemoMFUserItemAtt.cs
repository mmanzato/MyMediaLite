using System;
using System.Collections.Generic;
using System.Globalization;
using MyMediaLite.Data;
using MyMediaLite.Correlation;
using MyMediaLite.DataType;
using MyMediaLite.Taxonomy;
using System.Linq;

using System.IO;
using MyMediaLite.IO;


namespace MyMediaLite.RatingPrediction
{
	public class DemoMFUserItemAtt : DemoUserBaseline, ITransductiveRatingPredictor, IItemAttributeAwareRecommender
	{
		/// <summary>rating biases of the users</summary>
		protected internal float[] user_bias;
		/// <summary>rating biases of the items</summary>
		protected internal float[] item_bias;
		/// <summary>user factors (part expressed via the rated items)</summary>
		protected internal Matrix<float> y;
		/// <summary>user factors (individual part)</summary>
		protected internal Matrix<float> p;

		///
		public IDataSet AdditionalFeedback { get; set; }

		// TODO #332 update this structure on incremental updates
		/// <summary>The items rated by the users</summary>
		protected int[][] items_rated_by_user;
		/// <summary>precomputed regularization terms for the y matrix</summary>
		protected float[] y_reg;
		int[] feedback_count_by_item;

		/// <summary>bias learn rate</summary>
		public float BiasLearnRate { get; set; }
		/// <summary>regularization constant for biases</summary>
		public float BiasReg { get; set; }

		/// <summary>Regularization based on rating frequency</summary>
		/// <description>
		/// Regularization proportional to the inverse of the square root of the number of ratings associated with the user or item.
		/// As described in the paper by Menon and Elkan.
		/// </description>
		public bool FrequencyRegularization { get; set; }

		///
		public IBooleanMatrix ItemAttributes
		{
			get { return this.item_attributes; }
			set {
				this.item_attributes = value;
				this.NumItemAttributes = item_attributes.NumberOfColumns;
				this.MaxItemID = Math.Max(MaxItemID, item_attributes.NumberOfRows - 1);
			}
		}
		///
		protected IBooleanMatrix item_attributes;
		
		///
		public List<IBooleanMatrix> AdditionalItemAttributes
		{
			get { return this.additional_item_attributes; }
			set {
				this.additional_item_attributes = value;
			}
		}
		private List<IBooleanMatrix> additional_item_attributes;

		///
		public int NumItemAttributes { get; private set; }
		
		///
		protected Matrix<float>[] h;
		
		
		public DemoMFUserItemAtt () : base()
		{		
		}


				///
		public override void Train()
		{
			items_rated_by_user = this.ItemsRatedByUser();
			feedback_count_by_item = this.ItemFeedbackCounts();

			MaxUserID = Math.Max(MaxUserID, items_rated_by_user.Length - 1);
			MaxItemID = Math.Max(MaxItemID, feedback_count_by_item.Length - 1);

			y_reg = new float[MaxItemID + 1];
			for (int item_id = 0; item_id <= MaxItemID; item_id++)
				if (feedback_count_by_item[item_id] > 0)
					y_reg[item_id] = FrequencyRegularization ? (float) (Regularization / Math.Sqrt(feedback_count_by_item[item_id])) : Regularization;
				else
					y_reg[item_id] = 0;

			base.Train();
		}

		///
		protected override float Predict(int user_id, int item_id, bool bound)
		{
			double result = base.Predict(user_id, item_id, false);
						
			if (user_id < UserAttributes.NumberOfRows && item_id < ItemAttributes.NumberOfRows)
			{
				IList<int> item_attribute_list = ItemAttributes.GetEntriesByRow(item_id);
				double item_norm_denominator = item_attribute_list.Count;
				
				IList<int> user_attribute_list = UserAttributes.GetEntriesByRow(user_id);
				float user_norm_denominator = user_attribute_list.Count;
				
				float demo_spec = 0;
				float sum = 0;
				foreach(int u_att in user_attribute_list)
				{
					foreach(int i_att in item_attribute_list)
					{
						sum += h[0][u_att, i_att];
					}
				}				
				demo_spec += sum / user_norm_denominator;
				
				for(int d = 0; d < AdditionalUserAttributes.Count; d++)
				{
					user_attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(user_id);
					user_norm_denominator = user_attribute_list.Count;
					sum = 0;
					foreach(int u_att in user_attribute_list)
					{
						foreach(int i_att in item_attribute_list)
						{
							sum += h[d + 1][u_att, i_att];
						}
					}				
					demo_spec += sum / user_norm_denominator;
				}
				
				result += demo_spec / item_norm_denominator;
			}

			if (user_factors == null)
				PrecomputeUserFactors();
			if (user_id <= MaxUserID && item_id <= MaxItemID)
				result += DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);
			if (bound)
			{
				if (result > MaxRating)
					return MaxRating;
				if (result < MinRating)
					return MinRating;
			}
			return (float)result;
		}
		///
		protected internal override void InitModel()
		{
			base.InitModel();
			
			h = new Matrix<float>[AdditionalUserAttributes.Count + 1];
			h[0] = new Matrix<float>(UserAttributes.NumberOfColumns, ItemAttributes.NumberOfColumns);
			h[0].InitNormal(InitMean, InitStdDev);
			for(int d = 0; d < AdditionalUserAttributes.Count; d++)
			{
				h[d + 1] = new Matrix<float>(AdditionalUserAttributes[d].NumberOfColumns, ItemAttributes.NumberOfColumns);
				h[d + 1].InitNormal(InitMean, InitStdDev);
			}
			p = new Matrix<float>(MaxUserID + 1, NumFactors);
			p.InitNormal(InitMean, InitStdDev);
			y = new Matrix<float>(MaxItemID + 1, NumFactors);
			y.InitNormal(InitMean, InitStdDev);

			// set factors to zero for items without training examples
			for (int i = 0; i < ratings.CountByItem.Count; i++)
				if (ratings.CountByItem[i] == 0)
					y.SetRowToOneValue(i, 0);
			for (int i = ratings.CountByItem.Count; i <= MaxItemID; i++)
			{
				y.SetRowToOneValue(i, 0);
				item_factors.SetRowToOneValue(i, 0);
			}
			// set factors to zero for users without training examples (rest is done in MatrixFactorization.cs)
			for (int u = ratings.CountByUser.Count; u <= MaxUserID; u++)
				p.SetRowToOneValue(u, 0);

			user_bias = new float[MaxUserID + 1];
			item_bias = new float[MaxItemID + 1];
		}
		
		///
		protected override void Iterate(IList<int> rating_indices, bool update_user, bool update_item)
		{
			user_factors = null; // delete old user factors
			float reg = Regularization; // to limit property accesses			
			
			foreach (int index in rating_indices)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];
				float prediction = global_bias + user_bias[u] + item_bias[i];
				var p_plus_y_sum_vector = y.SumOfRows(items_rated_by_user[u]);
				double norm_denominator = Math.Sqrt(items_rated_by_user[u].Length);
				prediction += DataType.MatrixExtensions.RowScalarProduct(item_factors, i, p_plus_y_sum_vector);

				float err = ratings[index] - prediction;
				
				float user_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByUser[u])) : reg;
				float item_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByItem[i])) : reg;

				// adjust biases
				if (update_user)
					user_bias[u] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * user_reg_weight * user_bias[u]);
				if (update_item)
					item_bias[i] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * item_reg_weight * item_bias[i]);

				// adjust demo global attributes
				if(u < UserAttributes.NumberOfRows)
				{
					IList<int> attribute_list = UserAttributes.GetEntriesByRow(u);
					if(attribute_list.Count > 0)
					{
						foreach (int attribute_id in attribute_list)
						{							
							main_demo[attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * main_demo[attribute_id]);
						}
					}
				}
				
				for(int d = 0; d < AdditionalUserAttributes.Count; d++)
				{
					if(u < AdditionalUserAttributes[d].NumberOfRows)
					{
						IList<int> attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(u);
						if(attribute_list.Count > 0)
						{
							foreach (int attribute_id in attribute_list)
							{
								second_demo[d][attribute_id] += BiasLearnRate * current_learnrate * (err - BiasReg * Regularization * second_demo[d][attribute_id]);								
							}
						}
					}	
				}
				
				// adjust demo specific attributes
				if(u < UserAttributes.NumberOfRows && i < ItemAttributes.NumberOfRows)
				{
					IList<int> item_attribute_list = ItemAttributes.GetEntriesByRow(i);
					float item_norm_denominator = item_attribute_list.Count;
					
					IList<int> user_attribute_list = UserAttributes.GetEntriesByRow(u);
					float user_norm = 1 / user_attribute_list.Count;				
					
					float norm_error = err / item_norm_denominator;
					
					foreach(int u_att in user_attribute_list)
					{
						foreach(int i_att in item_attribute_list)
						{
							h[0][u_att, i_att] += current_learnrate * (norm_error * user_norm - Regularization * h[0][u_att, i_att]);
						}
					}								
					
					for(int d = 0; d < AdditionalUserAttributes.Count; d++)
					{
						user_attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(u);
						user_norm = 1 / user_attribute_list.Count;
						
						foreach(int u_att in user_attribute_list)
						{
							foreach(int i_att in item_attribute_list)
							{
								h[d + 1][u_att, i_att] += current_learnrate * (norm_error * user_norm - Regularization * h[d + 1][u_att, i_att]);;
							}
						}									
					}
				}

				// adjust factors
				double normalized_error = err / norm_denominator;
				for (int f = 0; f < NumFactors; f++)
				{
					float i_f = item_factors[i, f];

					// if necessary, compute and apply updates
					if (update_user)
					{
						double delta_u = err * i_f - user_reg_weight * p[u, f];
						p.Inc(u, f, current_learnrate * delta_u);
					}
					if (update_item)
					{
						double delta_i = err * p_plus_y_sum_vector[f] - item_reg_weight * i_f;
						item_factors.Inc(i, f, current_learnrate * delta_i);
						double common_update = normalized_error * i_f;
						foreach (int other_item_id in items_rated_by_user[u])
						{
							double delta_oi = common_update - y_reg[other_item_id] * y[other_item_id, f];
							y.Inc(other_item_id, f, current_learnrate * delta_oi);
						}
					}
				}
			}

			UpdateLearnRate();
		}


		/// <summary>Precompute all user factors</summary>
		protected void PrecomputeUserFactors()
		{
			if (user_factors == null)
				user_factors = new Matrix<float>(MaxUserID + 1, NumFactors);

			if (items_rated_by_user == null)
				items_rated_by_user = this.ItemsRatedByUser();

			for (int user_id = 0; user_id <= MaxUserID; user_id++)
				PrecomputeUserFactors(user_id);
		}

		/// <summary>Precompute the factors for a given user</summary>
		/// <param name='user_id'>the ID of the user</param>
		protected virtual void PrecomputeUserFactors(int user_id)
		{
			if (items_rated_by_user[user_id].Length == 0)
				return;

			// compute
			var factors = y.SumOfRows(items_rated_by_user[user_id]);
			double norm_denominator = Math.Sqrt(items_rated_by_user[user_id].Length);
			for (int f = 0; f < factors.Count; f++)
				factors[f] = (float) (factors[f] / norm_denominator + p[user_id, f]);

			// assign
			for (int f = 0; f < factors.Count; f++)
				user_factors[user_id, f] = (float) factors[f];
		}
	}
}

