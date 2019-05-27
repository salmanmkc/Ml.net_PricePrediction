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
            input.Trip_distance = 2;
            input.Passenger_count = 1;
            input.Trip_time_in_secs = 1099;
            //input.Payment_type = "CRD":

            // Try model on sample data
            ModelOutput result = predEngine.Predict(input);

            Console.WriteLine(result.Score);


            /* model findings:
             * -- first is for 20s model, second linie will represent the more trained model with RS 0.9631 (600s), from RS 0.944 (20s)
              Distance: 3       Passenger Count: 3      Trip time (s): 1099     Price RS 0.944:     14.11699
              Distance: 3       Passenger Count: 1      Trip time (s): 1099     Price RS 0.944:     14.11699

              Distance: 2       Passenger Count: 1      Trip time (s): 1099     Price RS 0.944:     12.59949
                                                                                Price RS 0.9361:    12.41535        

              Distance: 2       Passenger Count: 1      Trip time (s): 800      Price RS 0.944:     10.5829   
                                                                                Price RS 0.9361:    10.63229
                                                                                
              Distance: 1.3f    Passenger Count: 1      Trip time (s): 385      Price RS 0.944, this is good since actual data has 1.3f, 1,385 as an output of 7
                                                                                Price RS 0.9361:  6.866996
                                                                              
              

             */
        }
    }
}



