# Karls.CaptchaReader

.NET library using the pretrained Onnx models from DDDDOCR to decode captchas.

The models under `src/DdddOCR.NET/OnnxModel` are taken from
<https://github.com/rhy3h/ddddocr-node> which in turn is based on
<https://github.com/sml2h3/ddddocr/>, both under the MIT license.

Utilizes the Onnx runtime for .NET to run the models.

## Installation

You can install the library via NuGet:

```bash
dotnet add package Karls.CaptchaReader
```

## Usage

```csharp
using Karls.CaptchaReader;

using var ocrReader = new DdddOcrReader();
var result = await ocrReader.ReadTextAsync("path/to/image.png", CancellationToken.None);

Console.WriteLine(result.Text);
```
