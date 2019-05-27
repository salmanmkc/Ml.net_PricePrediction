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
            input.Trip_distance = 3;
            input.Passenger_count = 3;
            input.Trip_time_in_secs = 1099;
            //input.Payment_type = "CRD":

            // Try model on sample data
            ModelOutput result = predEngine.Predict(input);

            Console.WriteLine(result.Score);


          /* model findings:
            Distance: 3     Passenger Count: 3      Trip time (s): 1099     Price:  14.11699
            



           */
        }
    }
}



