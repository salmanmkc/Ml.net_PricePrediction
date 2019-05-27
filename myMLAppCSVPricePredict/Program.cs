using System;
// Add ML.NET namespaces
using Microsoft.ML;
using MyMLAppCSVPricePredictML.Model.DataModels;

namespace myMLAppCSVPricePredict
{
    class Program
    {
        static void Main(string[] args)
        {
            ConsumeModel();
        }

        public static void ConsumeModel()
        {

            // Load the model
            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Use the code below to add input data
            var input = new ModelInput();
            input.Trip_distance = 1.3f;
            input.Passenger_count = 1;
            input.Trip_time_in_secs = 385;
            //input.Payment_type = "CRD":

            // Try model on sample data
            ModelOutput result = predEngine.Predict(input);

            Console.WriteLine(result.Score);


            /* model findings:
              Distance: 3     Passenger Count: 3      Trip time (s): 1099     Price:  14.11699
              Distance: 3     Passenger Count: 1      Trip time (s): 1099     Price:  14.11699
              Distance: 2     Passenger Count: 1      Trip time (s): 1099     Price:  12.59949
              Distance: 2     Passenger Count: 1      Trip time (s): 800      Price:  10.5829   
              Distance: 1.3f  Passenger Count: 1      Trip time (s): 385      Price:  6.830118, this is good since actual data has 1.3f, 1,385 as an output of 7
              

             */
        }
    }
}



