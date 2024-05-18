// Machine Learning Utils
// File name: Program.cs
// Code It Yourself with .NET, 2024

using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;

using MatrixBenchmark;

Summary summary = BenchmarkRunner.Run<TypedVsUntypedVsFlat>();

