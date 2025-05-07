using System.IO.Abstractions;
using System.Text;
using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DdddOCR.NET.Core;

public class OcrReader {
    private readonly MLContext _mlContext = new();
    private readonly IFileSystem _fileSystem = new FileSystem();
    private readonly string _modelLocation = GetAbsolutePath("OnnxModel/common_old.onnx");
    private readonly string _charsetLocation = GetAbsolutePath("OnnxModel/common_old.json");

    private static string GetAbsolutePath(string relativePath) {
#pragma warning disable IO0004 // Replace FileInfo class with IFileSystem.FileInfo for improved testability
        var dataRoot = new FileInfo(typeof(OcrReader).Assembly.Location);
#pragma warning restore IO0004 // Replace FileInfo class with IFileSystem.FileInfo for improved testability
        var assemblyFolderPath = dataRoot!.Directory!.FullName;

        var fileSystem = new FileSystem();
        var fullPath = fileSystem.Path.Combine(assemblyFolderPath, relativePath);

        return fullPath;
    }

    public string? ReadText(string imagePath) {
        var (imageData, width, height) = PreprocessImage(imagePath);
        var pipeline = _mlContext.Transforms.ApplyOnnxModel(
            outputColumnNames: ["387"],
            inputColumnNames: ["input1"],
            modelFile: _modelLocation
        );

        var emptyData = _mlContext.Data.LoadFromEnumerable(Array.Empty<OnnxInput>());
        var data = _mlContext.Data.LoadFromEnumerable([new OnnxInput { ImageData = imageData }]);

        var view = pipeline.Fit(emptyData).Transform(data);

        var output = view.GetColumn<float[]>("387").FirstOrDefault(); // [W:onnxruntime:, execution_frame.cc:876 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,-1} does not match actual shape of {30,1,8210} for output 387

        var dimensions = new[] { 30, 1, 8210 };

        var tensor = new DenseTensor<float>(output, dimensions);
        var argMaxValues = new int[dimensions[0]];
        for(var i = 0; i < dimensions[0]; i++) {
            var maxIndex = 0;
            var maxValue = float.MinValue;
            for(var j = 0; j < dimensions[2]; j++) {
                if(tensor[i, 0, j] > maxValue) {
                    maxValue = tensor[i, 0, j];
                    maxIndex = j;
                }
            }

            argMaxValues[i] = maxIndex;
        }

        return MapToCharset(argMaxValues);
    }

    private string MapToCharset(int[] argMaxValues) {
        var jsonData = _fileSystem.File.ReadAllText(_charsetLocation);
        var charset = JsonSerializer.Deserialize<string[]>(jsonData)!;
        var result = new StringBuilder();

        foreach(var value in argMaxValues) {
            if(value <= 0 || value >= charset.Length) {
                continue;
            }

            result.Append(charset[value]);
        }

        return result.ToString();
    }

    private (float[] imageData, int width, int height) PreprocessImage(string imagePath) {
        using var image = Image.Load<Rgba32>(imagePath);

        var targetHeight = 64;
        var targetWidth = (int)Math.Floor(image.Width * ((double)targetHeight / image.Height));
        var resizedImage = image.Clone(ctx => ctx
            .Resize(new ResizeOptions {
                Size = new Size(targetWidth, targetHeight),
                Mode = ResizeMode.Max
            })
            .Grayscale()
        );

        var imageData = new float[resizedImage.Width * resizedImage.Height];
        var memoryGroup = resizedImage.GetPixelMemoryGroup();
        foreach(var memory in memoryGroup) {
            if(memory.IsEmpty) {
                continue;
            }

            var span = memory.Span;
            for(var i = 0; i < span.Length; i++) {
                imageData[i] = (span[i].R / 255f) * 2 - 1;
            }
        }

        return (imageData, resizedImage.Width, resizedImage.Height);
    }
}

public class OnnxInput {
    [ColumnName("input1")]
    public float[] ImageData { get; set; } = [];
}

public class OnnxOutput {
    [ColumnName("data")]
    public float[] Data { get; set; } = [];

    [ColumnName("dims")]
    public int[] Dims { get; set; } = [];
}
