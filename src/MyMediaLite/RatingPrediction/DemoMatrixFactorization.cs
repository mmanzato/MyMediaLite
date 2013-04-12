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
	/// Demo matrix factorization.
	/// </summary>
	public class DemoMatrixFactorization : BiasedMatrixFactorization, IUserAttributeAwareRecommender
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
		
		
		public DemoMatrixFactorization () : base()
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
			SetupLoss();

			foreach (int index in rating_indices)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];

				/*double score = global_bias + user_bias[u] + item_bias[i] + DataType.MatrixExtensions.RowScalarProduct(user_factors, u, item_factors, i);
				double sig_score = 1 / (1 + Math.Exp(-score));

				double prediction = min_rating + sig_score * rating_range_size;*/
				
				double prediction = Predict(u, i);
				double sig_score = (prediction - min_rating) / rating_range_size;
				
				double err = ratings[index] - prediction;

				float gradient_common = compute_gradient_common(sig_score, err);

				float user_reg_weight = FrequencyRegularization ? (float) (RegU / Math.Sqrt(ratings.CountByUser[u])) : RegU;
				float item_reg_weight = FrequencyRegularization ? (float) (RegI / Math.Sqrt(ratings.CountByItem[i])) : RegI;

				// adjust biases
				if (update_user)
					user_bias[u] += BiasLearnRate * current_learnrate * (gradient_common - BiasReg * user_reg_weight * user_bias[u]);
				if (update_item)
					item_bias[i] += BiasLearnRate * current_learnrate * (gradient_common - BiasReg * item_reg_weight * item_bias[i]);
				
				// adjust attributes
				if(u < user_attributes.NumberOfRows)
				{
					IList<int> attribute_list = user_attributes.GetEntriesByRow(u);
					if(attribute_list.Count > 0)
					{
						foreach (int attribute_id in attribute_list)
						{							
							main_demo[attribute_id] += BiasLearnRate * current_learnrate * (gradient_common - BiasReg * Regularization * main_demo[attribute_id]);
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
							foreach (int attribute_id in attribute_list)
							{
								second_demo[d][attribute_id] += BiasLearnRate * current_learnrate * (gradient_common - BiasReg * Regularization * second_demo[d][attribute_id]);								
							}
						}
					}	
				}
				
				
				// adjust latent factors
				for (int f = 0; f < NumFactors; f++)
				{
					double u_f = user_factors[u, f];
					double i_f = item_factors[i, f];

					if (update_user)
					{
						double delta_u = gradient_common * i_f - user_reg_weight * u_f;
						user_factors.Inc(u, f, current_learnrate * delta_u);
						// this is faster (190 vs. 260 seconds per iteration on Netflix w/ k=30) than
						//    user_factors[u, f] += learn_rate * delta_u;
					}
					if (update_item)
					{
						double delta_i = gradient_common * u_f - item_reg_weight * i_f;
						item_factors.Inc(i, f, current_learnrate * delta_i);
					}
				}
			}
		}

		///
		public override float Predict(int user_id, int item_id)
		{
			double score = global_bias;

			if (user_id < user_bias.Length)
				score += user_bias[user_id];
			if (item_id < item_bias.Length)
				score += item_bias[item_id];
			if (user_id < user_factors.dim1 && item_id < item_factors.dim1)
				score += DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);
			
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
					score += sum / second_norm_denominator;
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
						score += sum / second_norm_denominator;
					}
				}	
			}
			
			return (float) (min_rating + ( 1 / (1 + Math.Exp(-score)) ) * rating_range_size);
		}
	}
}

