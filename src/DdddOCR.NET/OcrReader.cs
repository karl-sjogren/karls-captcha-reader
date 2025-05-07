using System.IO.Abstractions;
using System.Text;
using System.Text.Json;
using DdddOCR.NET.Extensions;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DdddOCR.NET;

public sealed class OcrReader : IDisposable {
    private readonly IFileSystem _fileSystem;
    private readonly ILogger<OcrReader> _logger;

    private readonly InferenceSession _session;

    private const string _modelLocation = "OnnxModel/common_old.onnx";
    private const string _charsetLocation = "OnnxModel/common_old.json";

    public OcrReader() {
        _fileSystem = new FileSystem();
        _logger = NullLoggerFactory.Instance.CreateLogger<OcrReader>();

        _session = new InferenceSession(GetAbsolutePath(_modelLocation), new SessionOptions {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        });
    }

    public OcrReader(IFileSystem fileSystem, ILogger<OcrReader> logger) {
        _fileSystem = fileSystem;
        _logger = logger;

        _session = new InferenceSession(GetAbsolutePath(_modelLocation), new SessionOptions {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        });
    }

    public void Dispose() {
        _session.Dispose();
    }

    private string GetAbsolutePath(string relativePath) {
        var dataRoot = _fileSystem.FileInfo.New(typeof(OcrReader).Assembly.Location);
        var assemblyFolderPath = dataRoot!.Directory!.FullName;

        var fileSystem = new FileSystem();
        var fullPath = fileSystem.Path.Combine(assemblyFolderPath, relativePath);

        return fullPath;
    }

    public async Task<string?> ReadTextAsync(string imagePath, CancellationToken cancellationToken) {
        var (imageData, width, height) = await PreprocessImageAsync(imagePath, cancellationToken);

        var inputTensor = new DenseTensor<float>(imageData, [1, 1, height, width]);

        using var result = _session.Run([
            NamedOnnxValue.CreateFromTensor("input1", inputTensor)
        ], _session.OutputNames);

        var output = result.FirstOrDefault();

        if(output == null) {
            _logger.LogError("No output found in the model result.");
            return null;
        }

        var tensor = output.AsTensor<float>();

        var argMaxValues = tensor.ArgMax();

        return await MapToCharsetAsync(argMaxValues);
    }

    internal async Task<string> MapToCharsetAsync(int[] argMaxValues) {
        var jsonData = await _fileSystem.File.ReadAllTextAsync(GetAbsolutePath(_charsetLocation));
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

    private async Task<(float[] imageData, int width, int height)> PreprocessImageAsync(string imagePath, CancellationToken cancellationToken) {
        using var image = await Image.LoadAsync<Rgba32>(imagePath, cancellationToken);

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
                imageData[i] = ((span[i].R / 255f) * 2) - 1;
            }
        }

        return (imageData, resizedImage.Width, resizedImage.Height);
    }
}
