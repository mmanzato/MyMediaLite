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
using MyMediaLite.Correlation;
using MyMediaLite.Data;
using MyMediaLite.DataType;
using MyMediaLite.IO;

namespace MyMediaLite.ItemRecommendation
{
	/// <summary>
	/// Demo user similarity.
	/// </summary>
	public class DemoUserSimilarity : MF, IUserAttributeAwareRecommender
	{
		/// <summary>
		/// Initializes a new instance of the <see cref="MyMediaLite.RatingPrediction.DemoUserSimilarity"/> class.
		/// </summary>
		public DemoUserSimilarity () : base()
		{
			StopCondition = 0.0001;
			alpha = 0.1f;
		}
		
		/// <summary>Stop Condition</summary>
		public double StopCondition { get; set; }
		
		/// <summary>Alpha parameter</summary>
		public float alpha { get; set; }
		
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
		
		/// <summary>Matrix containing the user gradients</summary>
		protected internal Matrix<float> user_gradients;

		/// <summary>Matrix containing the item gradients</summary>
		protected internal Matrix<float> item_gradients;
		
		/// <summary>Matrix containing demographic-based users similarities</summary>
		//protected internal Matrix<float> user_demo_spec;
		
		/// <summary>Correlation matrix over some kind of entity</summary>
		protected ICorrelationMatrix correlation;
		
		///
		protected IBooleanMatrix BinaryDataMatrix { get { return user_attributes; } }
		
		///
		private Matrix<float> aux_user_factors;
		
		///
		private Matrix<float> aux_item_factors;	
		
		///
		private float cost_t;
		
		///
		private float learning_rate;
		
		///
		private Boolean done;
		
		/// <summary>Initialize the model data structure</summary>
		protected override void InitModel()
		{
			base.InitModel();			
			
			correlation = new BinaryCosine(MaxUserID + 1);			
			((IBinaryDataCorrelationMatrix) correlation).ComputeCorrelations(BinaryDataMatrix);
			
			// init gradients
			user_gradients = new Matrix<float>(MaxUserID + 1, NumFactors);
			item_gradients = new Matrix<float>(MaxItemID + 1, NumFactors);
			
			aux_user_factors = new Matrix<float>(MaxUserID + 1, NumFactors);
			aux_item_factors = new Matrix<float>(MaxItemID + 1, NumFactors);			
			
			global_bias = ratings.Average;
			
			cost_t = ComputeLoss(user_factors, item_factors);
			
			learning_rate = 0.1f;
			
			done = false;
		}
		
		///
		public override void Iterate()
		{
			if(!done)
			{
				Console.WriteLine("cost_t: " + cost_t);
				learning_rate *= 2;
				ComputeGradients();
				
				//aux_user_factors.Init(0);
				//aux_item_factors.Init(0);				
				for(int u = 0; u < MaxUserID + 1; u++)
				{
					for(int f = 0; f < NumFactors; f++)
					{
						aux_user_factors[u, f] = user_factors[u, f] - learning_rate * user_gradients[u, f];
					}
				}
				for(int i = 0; i < MaxItemID + 1; i++)
				{
					for(int f = 0; f < NumFactors; f++)
					{
						aux_item_factors[i, f] = item_factors[i, f] - learning_rate * item_gradients[i, f];
					}
				}
				float cost_aux = ComputeLoss(aux_user_factors, aux_item_factors);
				
				while(cost_aux >= cost_t)
				{
					learning_rate /= 2f;
					//aux_user_factors.Init(0);
					//aux_item_factors.Init(0);
					for(int u = 0; u < MaxUserID + 1; u++)
					{
						for(int f = 0; f < NumFactors; f++)
						{
							aux_user_factors[u, f] = user_factors[u, f] - learning_rate * user_gradients[u, f];
						}
					}
					for(int i = 0; i < MaxItemID + 1; i++)
					{
						for(int f = 0; f < NumFactors; f++)
						{
							aux_item_factors[i, f] = item_factors[i, f] - learning_rate * item_gradients[i, f];
						}
					}
					cost_aux = ComputeLoss(aux_user_factors, aux_item_factors);
					Console.WriteLine("cost_aux: " + cost_aux);
				}
				
				//Console.WriteLine("Updating factors...");
				for(int u = 0; u < MaxUserID + 1; u++)
				{
					for(int f = 0; f < NumFactors; f++)
					{
						//if(u ==0)
						//	Console.WriteLine("Updating user_factors[0, {0}] = {1} - {2} * {3}", f, user_factors[u, f], learning_rate, user_gradients[u, f]);
						user_factors[u, f] = aux_user_factors[u, f];
					}
				}
				for(int i = 0; i < MaxItemID + 1; i++)
				{
					for(int f = 0; f < NumFactors; f++)
					{
						item_factors[i, f] = aux_item_factors[i, f];
					}
				}
				float delta = (cost_t - cost_aux) / cost_t;
				Console.WriteLine("delta: " + delta);
				
				if(delta <= StopCondition)
				{
					done = true;
				}
				else 
				{
					cost_t = cost_aux;
				}
			}
		}
		
		/// <summary>Predict the rating of a given user for a given item</summary>
		/// <remarks>
		/// If the user or the item are not known to the recommender, the global average is returned.
		/// To avoid this behavior for unknown entities, use CanPredict() to check before.
		/// </remarks>
		/// <param name="user_id">the user ID</param>
		/// <param name="item_id">the item ID</param>
		/// <returns>the predicted rating</returns>
		public override float Predict(int user_id, int item_id)
		{
			if (user_id >= user_factors.dim1)
				return global_bias;
			if (item_id >= item_factors.dim1)
				return global_bias;

			return DataType.MatrixExtensions.RowScalarProduct(user_factors, user_id, item_factors, item_id);
		}
		
		private float ComputeLoss(Matrix<float> user_factors, Matrix<float> item_factors)
		{
			float first_sum = 0;
			for (int index = 0; index < ratings.Count; index++)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];
				if (u <= MaxUserID && i <= MaxItemID)
				{
					float err = (ratings[index] - DataType.MatrixExtensions.RowScalarProduct(user_factors, u, item_factors, i));
					first_sum += err * err;
				}
			}			
			first_sum *= 0.5f;
			
			float second_sum = 0;
			IList<int> user_list = ratings.AllUsers;
			//for(int udx = 0; udx < user_list.Count - 1; udx++)
			foreach(int u in user_list)
			{
				foreach(int v in user_list)
				{
					if(u == v)
						continue;
					//float err = (correlation[user_list[udx], user_list[vdx]] - DataType.MatrixExtensions.RowScalarProduct(user_factors, user_list[udx], user_factors, user_list[vdx]));
					float err = (correlation[u, v] - DataType.MatrixExtensions.RowScalarProduct(user_factors, u, user_factors, v));
					second_sum += err * err;
				}
			}
			second_sum *= 0.5f;
			
			float user_frob = user_factors.FrobeniusNorm();
			float item_frob = item_factors.FrobeniusNorm();
			
			float result;
			result = first_sum + alpha * second_sum + 0.5f * Regularization * ((float)Math.Pow(user_frob, 2) + (float)Math.Pow(item_frob, 2));
			
			return result;
		}
		
		private void ComputeGradients()
		{
			//Console.WriteLine("Computing user gradients...");		
			user_gradients.Init(0);
			item_gradients.Init(0);
			IList<int> user_list = ratings.AllUsers;
			for(int udx = 0; udx < user_list.Count; udx++)
			{
				int u = user_list[udx];
				foreach(int index in ratings.ByUser[u])
				{
					int i = ratings.Items[index];
					float err = (DataType.MatrixExtensions.RowScalarProduct(user_factors, u, item_factors, i) - ratings[index]);
					var item_vector = item_factors.GetRow(i);
					for(int f = 0; f < item_vector.Count; f++)
					{
						user_gradients[u, f] = (item_vector[f] * err);
						//user_gradients[u, f] = (item_vector[f] * err) + Regularization * user_factors[u, f];
					}
				}
				//for(int vdx = udx + 1; vdx < user_list.Count; vdx++)
				for(int vdx = 0; vdx < user_list.Count; vdx++)
				{
					int v = user_list[vdx];
					if(u == v)
						continue;
					float err = (DataType.MatrixExtensions.RowScalarProduct(user_factors, u, user_factors, v) - correlation[u, v]);
					var user_vector = user_factors.GetRow(v);
					for(int f = 0; f < user_vector.Count; f++)
					{
						user_gradients[u, f] += (2 * alpha * user_vector[f] * err) + Regularization * user_factors[u, f];
					}
				}
			}
			//Console.WriteLine("Computing item gradients...");			
			IList<int> item_list = ratings.AllItems;
			for(int idx = 0; idx < item_list.Count; idx++)
			{
				int i = item_list[idx];
				foreach(int index in ratings.ByItem[i])
				{
					int u = ratings.Users[index];
					float err = (DataType.MatrixExtensions.RowScalarProduct(user_factors, u, item_factors, i) - ratings[index]);
					var user_vector = user_factors.GetRow(u);
					for(int f = 0; f < user_vector.Count; f++)
					{
						item_gradients[i, f] = (user_vector[f] * err) + Regularization * item_factors[i, f];
					}
				}
			}	
		}
		
		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"{0} num_factors={1} regularization={2} StopCondition={3} alpha={4}",
				this.GetType().Name, NumFactors, Regularization, StopCondition, alpha);
		}
	}
}

