using Microsoft.ML.OnnxRuntime.Tensors;

namespace Karls.CaptchaReader.Extensions;

internal static class TensorExtensions {
    internal static int[] ArgMax(this Tensor<float> tensor) {
        var dimensions = tensor.Dimensions;
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

        return argMaxValues;
    }
}
