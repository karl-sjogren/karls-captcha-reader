using System.IO.Abstractions;
using System.Text;
using Karls.CaptchaReader.Extensions;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Karls.CaptchaReader;

/// <summary>
/// <inheritdoc />
/// </summary>
public sealed class DdddOcrReader : IOcrReader, IDisposable {
    private readonly IFileSystem _fileSystem;
    private readonly ILogger<DdddOcrReader> _logger;

    private InferenceSession? _session;

    private const string _modelLocation = "./OnnxModel/common_old.onnx";

    /// <summary>
    /// Initializes a new instance of the <see cref="DdddOcrReader"/> class.
    /// </summary>
    public DdddOcrReader() {
        _fileSystem = new FileSystem();
        _logger = NullLoggerFactory.Instance.CreateLogger<DdddOcrReader>();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DdddOcrReader"/> class with the specified file system and logger.
    /// </summary>
    public DdddOcrReader(IFileSystem fileSystem, ILogger<DdddOcrReader> logger) {
        _fileSystem = fileSystem;
        _logger = logger;
    }

    public void Dispose() {
        _session?.Dispose();
    }

    private string GetAbsolutePath(string relativePath) {
        var dataRoot = _fileSystem.FileInfo.New(typeof(DdddOcrReader).Assembly.Location);
        var assemblyFolderPath = dataRoot!.Directory!.FullName;

        var fileSystem = new FileSystem();
        var fullPath = fileSystem.Path.Combine(assemblyFolderPath, relativePath);

        return fullPath;
    }

    private async Task<InferenceSession> GetSessionAsync(CancellationToken cancellationToken) {
        if(_session != null) {
            return _session;
        }

        var modelPath = GetAbsolutePath(_modelLocation);
        if(!_fileSystem.File.Exists(modelPath)) {
            throw new FileNotFoundException("Model file not found.", modelPath);
        }

        var modelBytes = await _fileSystem.File.ReadAllBytesAsync(modelPath, cancellationToken);
        if(modelBytes.Length == 0) {
            throw new InvalidOperationException("Model file is empty.");
        }

        var sessionOptions = new SessionOptions {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        };

        _session = new InferenceSession(modelBytes, sessionOptions);

        return _session;
    }

    public async Task<string?> ReadTextAsync(string imagePath, CancellationToken cancellationToken) {
        if(!_fileSystem.File.Exists(imagePath)) {
            throw new FileNotFoundException("Image file not found.", imagePath);
        }

        var buffer = await _fileSystem.File.ReadAllBytesAsync(imagePath, cancellationToken);
        if(buffer.Length == 0) {
            throw new InvalidOperationException("Image file is empty.");
        }

        return await ReadTextAsync(buffer, CancellationToken.None);
    }

    /// <summary>
    /// <inheritdoc />
    /// </summary>
    public async Task<string?> ReadTextAsync(byte[] imageBytes, CancellationToken cancellationToken) {
        var (imageData, width, height) = PreprocessImage(imageBytes);

        var inputTensor = new DenseTensor<float>(imageData, [1, 1, height, width]);

        var session = await GetSessionAsync(cancellationToken);
        using var result = session.Run([
            NamedOnnxValue.CreateFromTensor("input1", inputTensor)
        ], session.OutputNames);

        var output = result[0];

        if(output == null) {
            _logger.LogError("No output found in the model result.");
            return null;
        }

        var tensor = output.AsTensor<float>();

        var argMaxValues = tensor.ArgMax();

        return MapToCharset(argMaxValues);
    }

    internal static string MapToCharset(int[] argMaxValues) {
        var charset = DdddOcrCharset.Characters;
        var result = new StringBuilder();

        foreach(var value in argMaxValues) {
            if(value <= 0 || value >= charset.Count) {
                continue;
            }

            result.Append(charset[value]);
        }

        return result.ToString();
    }

    internal static (float[] imageData, int width, int height) PreprocessImage(byte[] buffer) {
        using var image = Image.Load<Rgba32>(buffer);

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
                imageData[i] = span[i].R / 255f * 2 - 1;
            }
        }

        return (imageData, resizedImage.Width, resizedImage.Height);
    }
}
