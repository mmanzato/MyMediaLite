// Copyright (C) 2012 Zeno Gantner
// 
// This file is part of MyMediaLite.
// 
// MyMediaLite is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// MyMediaLite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
//  You should have received a copy of the GNU General Public License
//  along with MyMediaLite.  If not, see <http://www.gnu.org/licenses/>.
// 
using System;
using System.Collections.Generic;
using System.Globalization;
using MyMediaLite.Data;
using MyMediaLite.Correlation;
using MyMediaLite.DataType;
using MyMediaLite.Taxonomy;
using System.Linq;

namespace MyMediaLite.RatingPrediction
{
	public class DemoMFUserKNN2 : DemoUserBaseline
	{
		/// <summary>Weights in the neighborhood model that represent coefficients relating items based on the existing ratings</summary>
		protected Matrix<float> w;
		
		/// <summary>Correlation matrix over some kind of entity</summary>
		protected ICorrelationMatrix correlation;
		
		///
		protected IList<int>[] k_relevant_users;
		
		/// <summary>Matrix indicating which user rated a given item</summary>
		protected SparseBooleanMatrix data_user;
		
		///
		protected IList<int> rkui;
		
		///
		public override IRatings Ratings
		{
			set {
				base.Ratings = value;

				data_user = new SparseBooleanMatrix();
				for (int index = 0; index < ratings.Count; index++)
					data_user[ratings.Users[index], ratings.Items[index]] = true;
			}
		}
		
		/// <summary>Number of neighbors to take into account for predictions</summary>
		public uint K { get { return k; } set { k = value; } }
		private uint k;
		
		public DemoMFUserKNN2 () : base()
		{
			K = 30;
		}
		
		///
		protected internal override void InitModel()
		{
			base.InitModel();

			correlation = new Pearson(MaxUserID + 1, 0);
			((IRatingCorrelationMatrix) correlation).ComputeCorrelations(Ratings, EntityType.USER);

			k_relevant_users = new IList<int>[MaxUserID + 1];
 			for (int user_id = 0; user_id <= MaxUserID; user_id++)
			{
				k_relevant_users[user_id] = correlation.GetNearestNeighbors(user_id, K);
			}

			rkui = new List<int>();

			w = new Matrix<float>(MaxUserID + 1, MaxUserID + 1);
			w.InitNormal(InitMean, InitStdDev);
		}
		
		///
		protected void UpdateSimilarUsers(int user_id, int item_id)
		{
			rkui.Clear();
			
			IList<int>[] u_att_list = new IList<int>[AdditionalUserAttributes.Count + 1];
				
			for(int d = 0; d < AdditionalUserAttributes.Count + 1; d++)
			{
				if(d == 0)
					u_att_list[d] = UserAttributes.GetEntriesByRow(user_id);
				else
					u_att_list[d] = AdditionalUserAttributes[d - 1].GetEntriesByRow(user_id);
			}
			
			foreach (int v in k_relevant_users[user_id]) 
			{
				IList<int>[] v_att_list = new IList<int>[AdditionalUserAttributes.Count + 1];
				
				for(int d = 0; d < AdditionalUserAttributes.Count + 1; d++)
				{
					if(d == 0)
						v_att_list[d] = UserAttributes.GetEntriesByRow(v);
					else
						v_att_list[d] = AdditionalUserAttributes[d - 1].GetEntriesByRow(v);
				}	
				
				if (data_user[v, item_id])
				{
					for(int d = 0; d < AdditionalUserAttributes.Count + 1; d++)
					{
						if(u_att_list[d].SequenceEqual(v_att_list[d]))
						{
							if(!rkui.Contains(v))
								rkui.Add(v);
						}	
					}
				}		
			}
		}
		
		///
		protected override void Iterate(IList<int> rating_indices, bool update_user, bool update_item)
		{
			float reg = Regularization; // to limit property accesses			
			
			foreach (int index in rating_indices)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];

				UpdateSimilarUsers(u, i);

				float prediction = Predict(u, i, false);
				float err = ratings[index] - prediction;
				
				float user_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByUser[u])) : reg;
				float item_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByItem[i])) : reg;

				// adjust biases
				if (update_user)
					user_bias[u] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * user_reg_weight * user_bias[u]);
				if (update_item)
					item_bias[i] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * item_reg_weight * item_bias[i]);
				
				// adjust attributes
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
				
				// adjust users similarities
				foreach (int v in rkui) 
				{					
					float rating  = ratings.Get(v, i, ratings.ByItem[i]);
					w[u, v] += current_learnrate * ((err / (float)Math.Sqrt(rkui.Count)) * (rating - BasePredict(v, i)) - reg * w[u, v]);		
					//w[u, v] += current_learnrate * ((err / (float)rkui.Count) * (rating - BasePredict(v, i)) - reg * w[u, v]);		
				}	
				
				// adjust latent factors
				/*for (int f = 0; f < NumFactors; f++)
				{
					double u_f = user_factors[u, f];
					double i_f = item_factors[i, f];

					if (update_user)
					{
						double delta_u = err * i_f - user_reg_weight * u_f;
						user_factors.Inc(u, f, current_learnrate * delta_u);
					}
					if (update_item)
					{
						double delta_i = err * u_f - item_reg_weight * i_f;
						item_factors.Inc(i, f, current_learnrate * delta_i);
					}
				}*/
			}

			UpdateLearnRate();
		}

		///
		protected float BasePredict(int user_id, int item_id)
		{
			double result =	global_bias + 
				 	user_bias[user_id] + 
					item_bias[item_id];/* + 
					DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);*/
			
			if(user_id < UserAttributes.NumberOfRows)
			{
				IList<int> attribute_list = UserAttributes.GetEntriesByRow(user_id);
				if(attribute_list.Count > 0)
				{
					double sum = 0;
					double second_norm_denominator = attribute_list.Count;
					foreach(int attribute_id in attribute_list) 
					{
						sum += main_demo[attribute_id];
					}
					result += sum / second_norm_denominator;
				}
			}
			
			for(int d = 0; d < AdditionalUserAttributes.Count; d++)
			{
				if(user_id < AdditionalUserAttributes[d].NumberOfRows)
				{
					IList<int> attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(user_id);
					if(attribute_list.Count > 0)
					{
						double sum = 0;
						double second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += second_demo[d][attribute_id];
						}
						result += sum / second_norm_denominator;
					}
				}	
			}
			
			return (float) result;
		}
		
		///
		protected override float Predict(int user_id, int item_id, bool bound)
		{						
			double result = global_bias;

			if (user_id < user_bias.Length)
				result += user_bias[user_id];
			if (item_id < item_bias.Length)
				result += item_bias[item_id];
			/*if (user_id < user_factors.dim1 && item_id < item_factors.dim1)
				result += DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);*/
			
			if(user_id < UserAttributes.NumberOfRows)
			{
				IList<int> attribute_list = UserAttributes.GetEntriesByRow(user_id);
				if(attribute_list.Count > 0)
				{
					double sum = 0;
					double second_norm_denominator = attribute_list.Count;
					foreach(int attribute_id in attribute_list) 
					{
						sum += main_demo[attribute_id];
					}
					result += sum / second_norm_denominator;
				}
			}
			
			for(int d = 0; d < AdditionalUserAttributes.Count; d++)
			{
				if(user_id < AdditionalUserAttributes[d].NumberOfRows)
				{
					IList<int> attribute_list = AdditionalUserAttributes[d].GetEntriesByRow(user_id);
					if(attribute_list.Count > 0)
					{
						double sum = 0;
						double second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += second_demo[d][attribute_id];
						}
						result += sum / second_norm_denominator;
					}
				}	
			}
			
			
			if (user_id <= MaxUserID && item_id <= MaxItemID)
			{
				float r_sum = 0;
				int r_count = 0;				

				if(bound)
				{
					UpdateSimilarUsers(user_id, item_id);	
				}

				foreach (int v in rkui) 
				{
					float rating  = ratings.Get(v, item_id, ratings.ByItem[item_id]);
					r_sum += (rating - BasePredict(v, item_id)) * w[user_id, v];
					r_count++;
				}

				if (r_count > 0)
					result += r_sum / (float)Math.Sqrt(r_count);				
					//result += r_sum / (float)r_count;
			}

			if (bound)
			{
				if (result > MaxRating)
					return MaxRating;
				if (result < MinRating)
					return MinRating;
			}
			return (float)result;
		}

		/// <summary>Predict the rating of a given user for a given item</summary>		
		/// <param name="user_id">the user ID</param>
		/// <param name="item_id">the item ID</param>
		/// <returns>the predicted rating</returns>
		public override float Predict(int user_id, int item_id)
		{
			return Predict(user_id, item_id, true);
		}
		
		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"{0} bias_reg={1} reg_u={2} reg_i={3} frequency_regularization={4} learn_rate={5} bias_learn_rate={6} learn_rate_decay={7} num_iter={8} K={9}",
				this.GetType().Name, BiasReg, RegU, RegI, FrequencyRegularization, LearnRate, BiasLearnRate, Decay, NumIter, K);
		}
	}
}

