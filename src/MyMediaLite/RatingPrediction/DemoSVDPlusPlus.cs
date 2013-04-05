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
using System.IO;
using System.Linq;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MyMediaLite.IO;

namespace MyMediaLite.RatingPrediction
{
	/// <summary>
	/// Demographic-based gSVD++.
	/// </summary>
	public class DemoSVDPlusPlus : GSVDPlusPlus, IUserAttributeAwareRecommender
	{
		///
		public IBooleanMatrix UserAttributes
		{
			get { return this.user_attributes; }
			set {
				this.user_attributes = value;
				this.NumUserAttributes = user_attributes.NumberOfColumns;
				this.MaxUserID = Math.Max(MaxUserID, user_attributes.NumberOfRows - 1);
			}
		}
		private IBooleanMatrix user_attributes;
		
		///
		public List<IBooleanMatrix> AdditionalUserAttributes
		{
			get { return this.additional_user_attributes; }
			set {
				this.additional_user_attributes = value;
			}
		}
		private List<IBooleanMatrix> additional_user_attributes;
		
		///
		public int NumUserAttributes { get; private set; }
		
		/// <summary>Main demographic biases</summary>
		protected float[] main_demo;
		
		/// <summary>Secondary biases</summary>
		protected List<float[]> second_demo;
		
		/// <summary>
		/// Initializes a new instance of the <see cref="MyMediaLite.RatingPrediction.DemoSVDPlusPlus"/> class.
		/// </summary>
		public DemoSVDPlusPlus () : base()
		{
		}
		
		///
		protected internal override void InitModel()
		{
			base.InitModel();
			
			main_demo = new float[user_attributes.NumberOfColumns];
			second_demo = new List<float[]>(additional_user_attributes.Count);
			for(int d = 0; d < additional_user_attributes.Count; d++)
			{
				float[] element = new float[additional_user_attributes[d].NumberOfColumns];			
				second_demo.Add(element);
			}			
		}
		
		///
		protected override void Iterate(IList<int> rating_indices, bool update_user, bool update_item)
		{
			user_factors = null; // delete old user factors
			item_factors = null; // delete old item factors
			float reg = Regularization; // to limit property accesses

			foreach (int index in rating_indices)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];

				double prediction = global_bias + user_bias[u] + item_bias[i];
				var p_plus_y_sum_vector = y.SumOfRows(items_rated_by_user[u]);
				double norm_denominator = Math.Sqrt(items_rated_by_user[u].Length);
				for (int f = 0; f < p_plus_y_sum_vector.Count; f++)
					p_plus_y_sum_vector[f] = (float) (p_plus_y_sum_vector[f] / norm_denominator + p[u, f]);
				
				if(u < user_attributes.NumberOfRows)
				{
					IList<int> attribute_list = user_attributes.GetEntriesByRow(u);
					if(attribute_list.Count > 0)
					{
						double sum = 0;
						double second_norm_denominator = attribute_list.Count;
						foreach(int attribute_id in attribute_list) 
						{
							sum += main_demo[attribute_id];
						}
						prediction += sum / second_norm_denominator;
					}
				}
				
				for(int d = 0; d < additional_user_attributes.Count; d++)
				{
					if(u < additional_user_attributes[d].NumberOfRows)
					{
						IList<int> attribute_list = additional_user_attributes[d].GetEntriesByRow(u);
						if(attribute_list.Count > 0)
						{
							double sum = 0;
							double second_norm_denominator = attribute_list.Count;
							foreach(int attribute_id in attribute_list) 
							{
								sum += second_demo[d][attribute_id];
							}
							prediction += sum / second_norm_denominator;
						}
					}	
				}
				
				var q_plus_x_sum_vector = q.GetRow(i);
				
				if(i < item_attributes.NumberOfRows)
				{
					IList<int> attribute_list = item_attributes.GetEntriesByRow(i);
					if (attribute_list.Count > 0)
					{
						double second_norm_denominator = attribute_list.Count;
						var x_sum_vector = x.SumOfRows(attribute_list);
						for (int f = 0; f < x_sum_vector.Count; f++)
							q_plus_x_sum_vector[f] += (float) (x_sum_vector[f] / second_norm_denominator);
					}
				}

				prediction += DataType.VectorExtensions.ScalarProduct(q_plus_x_sum_vector, p_plus_y_sum_vector);

				double err = ratings[index] - prediction;

				float user_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByUser[u])) : reg;
				float item_reg_weight = FrequencyRegularization ? (float) (reg / Math.Sqrt(ratings.CountByItem[i])) : reg;

				// adjust biases
				if (update_user)
					user_bias[u] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * user_reg_weight * user_bias[u]);
				if (update_item)
					item_bias[i] += BiasLearnRate * current_learnrate * ((float) err - BiasReg * item_reg_weight * item_bias[i]);
				
				// adjust attributes
				if(u < user_attributes.NumberOfRows)
				{
					IList<int> attribute_list = user_attributes.GetEntriesByRow(u);
					if(attribute_list.Count > 0)
					{
						double second_norm_denominator = attribute_list.Count;
						double second_norm_error = err / second_norm_denominator;

						foreach (int attribute_id in attribute_list)
						{							
							main_demo[attribute_id] += BiasLearnRate * current_learnrate * ((float) second_norm_error - BiasReg * reg * main_demo[attribute_id]);
						}
					}
				}
				
				for(int d = 0; d < additional_user_attributes.Count; d++)
				{
					if(u < additional_user_attributes[d].NumberOfRows)
					{
						IList<int> attribute_list = additional_user_attributes[d].GetEntriesByRow(u);
						if(attribute_list.Count > 0)
						{
							double second_norm_denominator = attribute_list.Count;
							double second_norm_error = err / second_norm_denominator;
	
							foreach (int attribute_id in attribute_list)
							{							
								second_demo[d][attribute_id] += BiasLearnRate * current_learnrate * ((float) second_norm_error - BiasReg * reg * second_demo[d][attribute_id]);
							}
						}
					}	
				}
				
				double normalized_error = err / norm_denominator;
				for (int f = 0; f < NumFactors; f++)
				{
					float i_f = q_plus_x_sum_vector[f];

					// if necessary, compute and apply updates
					if (update_user)
					{
						double delta_u = err * i_f - user_reg_weight * p[u, f];
						p.Inc(u, f, current_learnrate * delta_u);
					}
					
					if (update_item)
					{
						double common_update = normalized_error * i_f;
						foreach (int other_item_id in items_rated_by_user[u])
						{
							double delta_oi = common_update - y_reg[other_item_id] * y[other_item_id, f];
							y.Inc(other_item_id, f, current_learnrate * delta_oi);
						}

						double delta_i = err * p_plus_y_sum_vector[f] - item_reg_weight * q[i, f];
						q.Inc(i, f, current_learnrate * delta_i);

						// adjust attributes
						if(i < item_attributes.NumberOfRows)
						{
							IList<int> attribute_list = item_attributes.GetEntriesByRow(i);
							if (attribute_list.Count > 0)
							{
								double second_norm_denominator = attribute_list.Count;
								double second_norm_error = err / second_norm_denominator;
	
								foreach (int attribute_id in attribute_list)
								{
									double delta_oi = second_norm_error * p_plus_y_sum_vector[f] - x_reg[attribute_id] * x[attribute_id, f];
									x.Inc(attribute_id, f, current_learnrate * delta_oi);
								}
							}
						}
					}
				}
			}

			UpdateLearnRate();
		}
		
		///
		public override float Predict(int user_id, int item_id)
		{
			double result = global_bias;

			if (user_factors == null)
				PrecomputeUserFactors();

			if (item_factors == null)
				PrecomputeItemFactors();

			if (user_id < user_bias.Length)
				result += user_bias[user_id];
			if (item_id < item_bias.Length)
				result += item_bias[item_id];
			
			if(user_id < user_attributes.NumberOfRows)
			{
				IList<int> attribute_list = user_attributes.GetEntriesByRow(user_id);
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
			
			for(int d = 0; d < additional_user_attributes.Count; d++)
			{
				if(user_id < additional_user_attributes[d].NumberOfRows)
				{
					IList<int> attribute_list = additional_user_attributes[d].GetEntriesByRow(user_id);
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
				result += DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);

			if (result > MaxRating)
				return MaxRating;
			if (result < MinRating)
				return MinRating;

			return (float) result;
		}
	}
}

