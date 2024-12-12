using System;
using System.Diagnostics;

public class RootSolvingMethod
{
    // Static Random object for random number generation
    static Random random = new Random();


    //Change the below function if you want to test other functions----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // Original polynomial function and its derivative
    static Func<double, double> polynomialFunction1 = x => 0.0074 * Math.Pow(x, 4) - 0.284 * Math.Pow(x, 3) + 3.355 * Math.Pow(x, 2) - 12.183 * x + 5;
    static Func<double, double> polynomialFunction1Prime = x => 4 * 0.0074 * Math.Pow(x, 3) - 3 * 0.284 * Math.Pow(x, 2) + 2 * 3.355 * x - 12.183;

    // 10 Additional Test Functions
    // 1. High-Degree Polynomial
    static Func<double, double> f1 = x => 0.001 * Math.Pow(x, 6) - 0.05 * Math.Pow(x, 5) + 0.5 * Math.Pow(x, 4) - 2 * Math.Pow(x, 3) + 3 * Math.Pow(x, 2) - x + 10;
    static Func<double, double> f1Prime = x => 0.006 * Math.Pow(x, 5) - 0.25 * Math.Pow(x, 4) + 2 * Math.Pow(x, 3) - 6 * Math.Pow(x, 2) + 6 * x - 1;

    // 2. Trigonometric-Linear Mix: sin(x) - x/10
    static Func<double, double> f2 = x => Math.Sin(x) - (x / 10.0);
    static Func<double, double> f2Prime = x => Math.Cos(x) - 0.1;

    // 3. Exponential: e^x - 5
    static Func<double, double> f3 = x => Math.Exp(x) - 5;
    static Func<double, double> f3Prime = x => Math.Exp(x);

    // 4. Complex Polynomial: x^5 - 5x^4 + 10x^2 - x + 1
    static Func<double, double> f4 = x => Math.Pow(x, 5) - 5 * Math.Pow(x, 4) + 10 * Math.Pow(x, 2) - x + 1;
    static Func<double, double> f4Prime = x => 5 * Math.Pow(x, 4) - 20 * Math.Pow(x, 3) + 20 * x - 1;

    // 5. Rational: (x^3 - 2x + 1)/(x^2 + 1)
    static Func<double, double> f5 = x => (Math.Pow(x, 3) - 2 * x + 1) / (x * x + 1);
    // Derivative (not simplified): 
    static Func<double, double> f5Prime = x =>
    {
        double numeratorPrime = (3 * x * x - 2) * (x * x + 1) - (x * x * x - 2 * x + 1) * (2 * x);
        double denominator = Math.Pow(x * x + 1, 2);
        return numeratorPrime / denominator;
    };

    // 6. Logistic-Type: (1/(1+e^-x)) - 0.5
    static Func<double, double> f6 = x => (1.0 / (1.0 + Math.Exp(-x))) - 0.5;
    static Func<double, double> f6Prime = x =>
    {
        double ex = Math.Exp(-x);
        return ex / Math.Pow(1 + ex, 2);
    };

    // 7. Higher-Degree Polynomial: 0.002x^7 - 0.01x^6 + 0.5x^3 - x + 2
    static Func<double, double> f7 = x => 0.002 * Math.Pow(x, 7) - 0.01 * Math.Pow(x, 6) + 0.5 * Math.Pow(x, 3) - x + 2;
    static Func<double, double> f7Prime = x => 0.014 * Math.Pow(x, 6) - 0.06 * Math.Pow(x, 5) + 1.5 * Math.Pow(x, 2) - 1;

    // 8. Trig-Polynomial Mix: x sin(x) - 0.5
    static Func<double, double> f8 = x => x * Math.Sin(x) - 0.5;
    static Func<double, double> f8Prime = x => Math.Sin(x) + x * Math.Cos(x);

    // 9. Exponential-Polynomial Mix: x^2 e^-x - 0.1
    static Func<double, double> f9 = x => Math.Pow(x, 2) * Math.Exp(-x) - 0.1;
    static Func<double, double> f9Prime = x => Math.Exp(-x) * (2 * x - x * x);

    // 10. Logarithmic-Quadratic Mix: ln(x) - x^2 + 3 (x>0)
    static Func<double, double> f10 = x => Math.Log(x) - x * x + 3;
    static Func<double, double> f10Prime = x => (1.0 / x) - 2 * x;

    // Flat derivative near x = 0
    static Func<double, double> flatDerivative = x => Math.Pow(x, 3);  // Root at x = 0
    static Func<double, double> flatDerivativePrime = x => 3 * Math.Pow(x, 2);

    // Piecewise function with a discontinuity at x = 0
    static Func<double, double> discontinuousFunction = x => x < 0 ? -1 : 1;  // No root
    static Func<double, double> discontinuousFunctionPrime = x => 0;  // Derivative undefined or zero

    // Trigonometric function with many roots
    static Func<double, double> highlyOscillatoryFunction = x => Math.Sin(10 * x);  // Roots at n * π/10
    static Func<double, double> highlyOscillatoryFunctionPrime = x => 10 * Math.Cos(10 * x);

    // Rational function with an asymptote
    static Func<double, double> rationalFunctionAsymptote = x => 1 / (x - 1);  // No root
    static Func<double, double> rationalFunctionAsymptotePrime = x => -1 / Math.Pow(x - 1, 2);

    // Exponential plateau near zero
    static Func<double, double> functionPlateau = x => Math.Exp(-x) - 0.1;  // Root near x ≈ -ln(0.1)
    static Func<double, double> functionPlateauPrime = x => -Math.Exp(-x);

    // Exponential decay function
    static Func<double, double> exponentialDecayFunction = x => Math.Exp(-x);  // No root
    static Func<double, double> exponentialDecayFunctionPrime = x => -Math.Exp(-x);

    public static void Main(string[] args)
    {

        // Define test cases: function name, f, fPrime, GA minX, GA maxX, bisection interval, newton starting guess
        // Adjust these intervals and starting guesses based on the nature of each function
        var testCases = new List<(string name, Func<double, double> f, Func<double, double> fPrime, double minX, double maxX, double startNewtonX)>
        {

            //To test one case edit polynomialFunction1 above--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            //Or uncomment all the below if you want to test all


            // Original test functions
            ("polynomialFunction1", polynomialFunction1, polynomialFunction1Prime, 15, 20, 17),
            //("f1", f1, f1Prime, -10, 10, 0),
            //("f2", f2, f2Prime, -2, 2, 0),
            //("f3", f3, f3Prime, -5, 5, 2),
            //("f4", f4, f4Prime, -5, 1, 1),
            //("f5", f5, f5Prime, -1, 1, 0),
            //("f6", f6, f6Prime, -10, 10, 0),
            //("f7", f7, f7Prime, -5, 5, 0),
            //("f8", f8, f8Prime, 0, 2, 1),
            //("f9", f9, f9Prime, 0, 5, 1),
            //("f10", f10, f10Prime, 1, 3, 1),

            //// Challenging functions for Newton-Raphson
            //("flatDerivative", flatDerivative, flatDerivativePrime, -1, 1, 0.1),
            //("discontinuousFunction", discontinuousFunction, discontinuousFunctionPrime, -1, 1, 0),
            //("highlyOscillatoryFunction", highlyOscillatoryFunction, highlyOscillatoryFunctionPrime, -.2, .2, 0),
            //("rationalFunctionAsymptote", rationalFunctionAsymptote, rationalFunctionAsymptotePrime, 0.5, 1.5, 0.9),
            //("functionPlateau", functionPlateau, functionPlateauPrime, -5, 5, -1),
            //("exponentialDecayFunction", exponentialDecayFunction, exponentialDecayFunctionPrime, -5, 5, 0)
         };

        List<(string funcName, double root, int iterations, double time)> geneticAlgorithmResults = new List<(string, double, int, double)>();
        List<(string funcName, double root, int iterations, double time)> bisectionResults = new List<(string, double, int, double)>();
        List<(string funcName, double root, int iterations, double time)> newtonRaphsonResults = new List<(string, double, int, double)>();

        Stopwatch sw = new Stopwatch();

        foreach (var (name, f, fPrime, minX, maxX, startX) in testCases)
        {
            Console.WriteLine($"Processing function: {name}");

            // Genetic Algorithm
            sw.Restart();

            //Edit these vars to adjust how the GeneticAlgorithm works------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            GeneticAlgorithm ga = new GeneticAlgorithm(
                f,
                populationSize: 100,
                crossoverRate: 0.7,
                mutationRate: 0.01,
                maxGenerations: 1000,
                minX: minX,
                maxX: maxX,
                tolerance: 1e-10 // Adjust tolerance as desired
            );

            (double gaRoot, int gaIterations) = ga.Run();
            sw.Stop();
            geneticAlgorithmResults.Add((name, gaRoot, gaIterations, sw.Elapsed.TotalMilliseconds));

            // Bisection
            sw.Restart();
            (double bisectRoot, int bisectIterations) = Bisection(minX, maxX, f);
            sw.Stop();
            bisectionResults.Add((name, bisectRoot, bisectIterations, sw.Elapsed.TotalMilliseconds));

            // Newton-Raphson
            sw.Restart();
            (double newtonRoot, int newtonIterations) = NewtonRaphson(startX, f, fPrime);
            sw.Stop();
            newtonRaphsonResults.Add((name, newtonRoot, newtonIterations, sw.Elapsed.TotalMilliseconds));

            Console.WriteLine($"Results for {name}:");
            Console.WriteLine($"  GA: root={gaRoot}, iterations={gaIterations}, time={geneticAlgorithmResults[^1].time} ms");
            Console.WriteLine($"  Bisection: root={bisectRoot}, iterations={bisectIterations}, time={bisectionResults[^1].time} ms");
            Console.WriteLine($"  Newton-Raphson: root={newtonRoot}, iterations={newtonIterations}, time={newtonRaphsonResults[^1].time} ms");
            Console.WriteLine();
        }

        // Output results to CSV files
        // GA Results
        using (var writer = new StreamWriter("ga_results.csv"))
        {
            writer.WriteLine("function,root,iterations,time_ms");
            foreach (var item in geneticAlgorithmResults)
            {
                writer.WriteLine($"{item.funcName},{item.root},{item.iterations},{item.time}");
            }
        }

        // Bisection Results
        using (var writer = new StreamWriter("bisection_results.csv"))
        {
            writer.WriteLine("function,root,iterations,time_ms");
            foreach (var item in bisectionResults)
            {
                writer.WriteLine($"{item.funcName},{item.root},{item.iterations},{item.time}");
            }
        }

        // Newton-Raphson Results
        using (var writer = new StreamWriter("newton_results.csv"))
        {
            writer.WriteLine("function,root,iterations,time_ms");
            foreach (var item in newtonRaphsonResults)
            {
                writer.WriteLine($"{item.funcName},{item.root},{item.iterations},{item.time}");
            }
        }

        Console.WriteLine("Results have been written to ga_results.csv, bisection_results.csv, and newton_results.csv.");
    }

    public static (double, int) Bisection(double xl, double xu, Func<double, double> polynomialFunction)
    {
        double xr = xl + Math.Abs(((xl - xu) / 2));
        double xrOld = 0;

        int i = 0;

        for (; i < 20 && Math.Abs((xr - xrOld) / xr) > 0.000000000000000000001; i++)
        {
            //Console.WriteLine("Number of iterations: " + i);
            double fl = polynomialFunction(xl);
            double fu = polynomialFunction(xu);
            double fr = polynomialFunction(xr);

            if (fr == 0)
            {
                return (xr, i);
            }
            else if (fl * fr < 0)
            {
                xu = xr;
            }
            else
            {
                xl = xr;
            }

            xrOld = xr;
            xr = xl + Math.Abs(((xl - xu) / 2));
        }

        return (xr, i);
    }

    public static (double, int) NewtonRaphson(double x, Func<double, double> polynomialFunction, Func<double, double> polynomialFunctionPrime)
    {
        double xOld = 0;
        int i = 0;

        do
        {
            //Console.WriteLine("Number of iterations: " + i);
            //Console.WriteLine("Current x: " + x);

            double fx = polynomialFunction(x);
            double fxPrime = polynomialFunctionPrime(x);


            xOld = x;

            x = x - fx / fxPrime;

            i++;

        } while (i < 20 && Math.Abs((x - xOld) / x) > 0.000000000000000000001);

        return (x, i);
    }


}

public class GeneticAlgorithm
{
    private Func<double, double> _function;
    private Random _random;

    // GA parameters
    private int _populationSize;
    private double _crossoverRate;
    private double _mutationRate;
    private int _maxGenerations;
    private double _minX;
    private double _maxX;
    private double _tolerance;

    // Population
    private List<Individual> _population;

    public GeneticAlgorithm(Func<double, double> function, int populationSize = 100, double crossoverRate = 0.7, double mutationRate = 0.01, int maxGenerations = 1000, double minX = -100, double maxX = 100, double tolerance = 1e-6)
    {
        _function = function;
        _populationSize = populationSize;
        _crossoverRate = crossoverRate;
        _mutationRate = mutationRate;
        _maxGenerations = maxGenerations;
        _minX = minX;
        _maxX = maxX;
        _tolerance = tolerance;

        _random = new Random();
        _population = new List<Individual>();
    }

    public (double, int) Run()
    {
        InitializePopulation();

        int generation = 0;

        for (; generation < _maxGenerations; generation++)
        {
            EvaluateFitness();

            // Check for solution
            _population.Sort((a, b) => b.Fitness.CompareTo(a.Fitness)); // Descending order
            if (Math.Abs(_function(_population[0].X)) < _tolerance)
            {
                // Found a solution
                Console.WriteLine($"Solution found at generation {generation}: x = {_population[0].X}, f(x) = {_function(_population[0].X)}");
                return (_population[0].X, generation);
            }

            // Perform selection to choose parents for reproduction
            List<Individual> selectedParents = Selection();

            // Create a new population
            List<Individual> newPopulation = new List<Individual>();

            // Elitism: Keep the best individual
            newPopulation.Add(selectedParents[0]);

            // Generate offspring to fill the rest of the new population
            while (newPopulation.Count < _populationSize)
            {
                // Randomly select two parents from the selected parents
                Individual parent1 = selectedParents[_random.Next(selectedParents.Count)];
                Individual parent2 = selectedParents[_random.Next(selectedParents.Count)];

                // Perform crossover to produce offspring
                Individual child1, child2;

                (child1, child2) = Crossover(parent1, parent2);

                // Apply mutation to offspring
                Mutate(child1);
                Mutate(child2);

                // Add offspring to the new population
                if (newPopulation.Count < _populationSize)
                    newPopulation.Add(child1);
                if (newPopulation.Count < _populationSize)
                    newPopulation.Add(child2);
            }

            // Update the population
            _population = newPopulation;
        }

        // Return best found solution after max generations
        _population.Sort((a, b) => b.Fitness.CompareTo(a.Fitness)); // Descending order
        Console.WriteLine($"No exact solution found. Best approximation: x = {_population[0].X}, f(x) = {_function(_population[0].X)}");
        return (_population[0].X, generation);
    }

    private void InitializePopulation()
    {
        _population.Clear();
        for (int i = 0; i < _populationSize; i++)
        {
            double x = _random.NextDouble() * (_maxX - _minX) + _minX;
            _population.Add(new Individual(x));
        }
    }

    private void EvaluateFitness()
    {
        foreach (var individual in _population)
        {
            double fx = _function(individual.X);
            individual.Fitness = 1.0 / (1.0 + Math.Abs(fx));
        }
    }

    private List<Individual> Selection()
    {
        // Order individuals by fitness (higher fitness first)
        var selectedIndividuals = _population
            .OrderByDescending(individual => individual.Fitness)
            .Take(_populationSize / 2)
            .ToList();
        return selectedIndividuals;
    }

    private (Individual, Individual) Crossover(Individual parent1, Individual parent2)
    {
        // Blend crossover
        double alpha = _random.NextDouble() * _crossoverRate;
        double child1X = alpha * parent1.X + (1 - alpha) * parent2.X;
        double child2X = (1 - alpha) * parent1.X + alpha * parent2.X;

        return (new Individual(child1X), new Individual(child2X));
    }

    private void Mutate(Individual individual)
    {
        if (_random.NextDouble() < _mutationRate)
        {
            // Small mutation
            double mutationAmount = (_maxX - _minX) * 0.01;
            double delta = (_random.NextDouble() * 2 - 1) * mutationAmount; // Random value between -mutationAmount and +mutationAmount
            individual.X += delta;

            // Ensure x is within bounds
            if (individual.X < _minX)
                individual.X = _minX;
            if (individual.X > _maxX)
                individual.X = _maxX;
        }
    }

    private class Individual
    {
        public double X { get; set; }
        public double Fitness { get; set; }

        public Individual(double x)
        {
            X = x;
        }
    }
}