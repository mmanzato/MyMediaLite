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

namespace MyMediaLite.RatingPrediction
{
	/// <summary>
	/// Demo user similarity.
	/// </summary>
	public class DemoUserSimilarity : MatrixFactorization, IUserAttributeAwareRecommender
	{
		/// <summary>
		/// Initializes a new instance of the <see cref="MyMediaLite.RatingPrediction.DemoUserSimilarity"/> class.
		/// </summary>
		public DemoUserSimilarity () : base()
		{
			StopCondition = 0.0001;
		}
		
		/// <summary>Stop Condition</summary>
		public double StopCondition { get; set; }
		
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
		
		/// <summary>Initialize the model data structure</summary>
		protected internal override void InitModel()
		{
			base.InitModel();			
			
			correlation = new BinaryCosine(MaxUserID + 1);
			
			// init gradients
			user_gradients = new Matrix<float>(MaxUserID + 1, NumFactors);
			item_gradients = new Matrix<float>(MaxItemID + 1, NumFactors);			
			user_gradients.Init(0);
			item_gradients.Init(0);
		}
		
		///
		public override void Train()
		{
			Console.WriteLine("Initing model...");
			InitModel();
			global_bias = ratings.Average;
			Console.WriteLine("Computing correlations...");
			((IBinaryDataCorrelationMatrix) correlation).ComputeCorrelations(BinaryDataMatrix);
			
			Console.Write("Computing Loss...");
			float cost_t = ComputeLoss(user_factors, item_factors);
			Console.WriteLine(cost_t);
			Boolean done = false;
			Matrix<float> aux_user_factors = new Matrix<float>(MaxUserID + 1, NumFactors);
			Matrix<float> aux_item_factors = new Matrix<float>(MaxItemID + 1, NumFactors);
			
			Console.WriteLine("Entering external loop...");
			do {
				Console.WriteLine("Iteraction...");
				float learning_rate = 1;
				ComputeGradients();
				float cost_aux = 0;
				do {
					learning_rate /= 2;
					
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
					Console.Write("Computing internal loss...");
					cost_aux = ComputeLoss(aux_user_factors, aux_item_factors);
					Console.WriteLine(cost_aux);
				} while (cost_aux >= cost_t);
				Console.WriteLine("Updating factors...");
				for(int u = 0; u < MaxUserID + 1; u++)
				{
					for(int f = 0; f < NumFactors; f++)
					{
						user_factors[u, f] -= learning_rate * user_gradients[u, f];
					}
				}
				for(int i = 0; i < MaxItemID + 1; i++)
				{
					for(int f = 0; f < NumFactors; f++)
					{
						item_factors[i, f] -= learning_rate * item_gradients[i, f];
					}
				}
				float new_cost = ComputeLoss(user_factors, item_factors);
				if(1 - new_cost / cost_t <= StopCondition)
				{
					done = true;	
				}
			} while(!done);
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
			float result = 0;
			for (int index = 0; index < ratings.Count; index++)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];
				if (u <= MaxUserID && i <= MaxItemID)
				{
					float err = (ratings[index] - DataType.MatrixExtensions.RowScalarProduct(user_factors, u, item_factors, i));
					result += err * err;
				}
			}
			IList<int> user_list = ratings.AllUsers;
			for(int u = 0; u < user_list.Count - 1; u++)
			{
				for(int v = u + 1; v < user_list.Count; v++)
				{
					float err = (correlation[user_list[u], user_list[v]] - DataType.MatrixExtensions.RowScalarProduct(user_factors, user_list[u], user_factors, user_list[v]));
					result += err * err;
				}
			}
			result += (Regularization / 2) * (user_factors.FrobeniusNorm() + item_factors.FrobeniusNorm());
			
			return result;
		}
		
		private void ComputeGradients()
		{
			for (int index = 0; index < ratings.Count; index++)
			{
				int u = ratings.Users[index];
				int i = ratings.Items[index];				
				if (u <= MaxUserID)
				{
					// compute gradient of user u
					IList<int> user_rating_indexes = ratings.ByUser[u];
					for(int index2 = 0; index2 < user_rating_indexes.Count; index2++)
					{
						int j = ratings.Items[index2];
						float err = (DataType.MatrixExtensions.RowScalarProduct(user_factors, u, item_factors, j) - ratings[index2]);
						var item_vector = item_factors.GetRow(j);
						for(int f = 0; f < item_vector.Count; f++)
						{
							user_gradients[u, f] = (item_vector[f] * err);
						}
					}
					IList<int> user_list = ratings.AllUsers;
					foreach(int v in user_list)
					{
						float err = (DataType.MatrixExtensions.RowScalarProduct(user_factors, u, user_factors, v) - correlation[u, v]);
						var user_vector = user_factors.GetRow(v);
						for(int f = 0; f < user_vector.Count; f++)
						{
							user_gradients[u, f] += (user_vector[f] * err) + Regularization * user_factors[u, f];
						}
					}
				}				
				if(i <= MaxItemID)
				{	
					// compute gradient of item i
					IList<int> item_rating_indexes = ratings.ByItem[i];
					for(int index2 = 0; index2 < item_rating_indexes.Count; index2++)
					{
						int v = ratings.Users[index2];
						float err = (DataType.MatrixExtensions.RowScalarProduct(user_factors, v, item_factors, i) - ratings[index2]);
						var user_vector = user_factors.GetRow(v);
						for(int f = 0; f < user_vector.Count; f++)
						{
							item_gradients[i, f] += (user_vector[f] * err) + Regularization * item_factors[i, f];
						}
					}
				}
			}	
		}
		
		///
		public override string ToString()
		{
			return string.Format(
				CultureInfo.InvariantCulture,
				"{0} num_factors={1} regularization={2} StopCondition={3}",
				this.GetType().Name, NumFactors, Regularization, StopCondition);
		}
	}
}

