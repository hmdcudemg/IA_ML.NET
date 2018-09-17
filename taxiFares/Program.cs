using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Threading.Tasks;

namespace taxiFares
{
    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "model.zip");

        static async Task Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFaresPrediction> model = await Train();
            Evaluate(model);

            TaxiTripFaresPrediction prediction = model.Predict(TestTrips.Trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFaresPrediction>> Train()
        {
            // This is the usual form to create this

            // var pipeline = new LearningPipeline();
            // pipeline.Add(new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','));
            // pipeline.Add(new ColumnCopier(("FareAmount")));
            // pipeline.Add(new CategoricalOneHotVectorizer("VendorId", "RateCode", "PaymentType"));
            // pipeline.Add(new ColumnConcatenator("Features","VendorId","RateCode","PassengerCount","TripDistance","PaymentType"));
            // pipeline.Add(new FastTreeRegressor());

            // This is the Handy form in C# to do this
            var pipeline = new LearningPipeline{
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),
                new ColumnCopier(("FareAmount","Label")),
                new CategoricalOneHotVectorizer("VendorId", "RateCode", "PaymentType"),
                new ColumnConcatenator("Features","VendorId","RateCode","PassengerCount","TripDistance","PaymentType"),
                new FastTreeRegressor()
            };

            PredictionModel<TaxiTrip, TaxiTripFaresPrediction> model = pipeline.Train<TaxiTrip, TaxiTripFaresPrediction>();

            await model.WriteAsync(_modelpath);
            return model;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFaresPrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }
    }
}
