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

            // Try model on sample data
            ModelOutput result = predEngine.Predict(input);


        }
    }
}



