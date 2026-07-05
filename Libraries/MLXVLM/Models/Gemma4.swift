import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Based on https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/gemma4

private enum Gemma4Error: LocalizedError {
    case imageTokenCountMismatch(expectedVisionTokens: Int, actualPromptTokens: Int)
    case imagePlaceholderMismatch(images: Int, placeholders: Int)

    var errorDescription: String? {
        switch self {
        case .imageTokenCountMismatch(let expectedVisionTokens, let actualPromptTokens):
            return
                "Gemma4 image token count mismatch: vision encoder produced \(expectedVisionTokens) soft tokens, but the prompt contains \(actualPromptTokens) image tokens."
        case .imagePlaceholderMismatch(let images, let placeholders):
            return
                "Gemma4 image placeholder mismatch: the request has \(images) images but the prompt contains at least \(placeholders) image placeholders."
        }
    }
}

private func gemma4BuildLayerTypes(hiddenLayers: Int, slidingWindowPattern: Int) -> [String] {
    let pattern =
        Array(repeating: "sliding_attention", count: max(slidingWindowPattern - 1, 0))
        + ["full_attention"]
    guard !pattern.isEmpty else { return Array(repeating: "full_attention", count: hiddenLayers) }
    var result: [String] = []
    result.reserveCapacity(hiddenLayers)
    while result.count < hiddenLayers {
        result.append(contentsOf: pattern)
    }
    return Array(result.prefix(hiddenLayers))
}

private func gemma4DefaultTextRopeParameters() -> [String: [String: StringOrNumber]] {
    [
        "full_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(1_000_000.0),
            "rope_type": .string("proportional"),
        ],
        "sliding_attention": [
            "partial_rotary_factor": .float(1.0),
            "rope_theta": .float(10_000.0),
            "rope_type": .string("default"),
        ],
    ]
}

private func gemma4DefaultVisionRopeParameters() -> [String: StringOrNumber] {
    [
        "rope_theta": .float(100.0),
        "rope_type": .string("default"),
    ]
}

private func gemma4MaskedScatter(
    inputTensor: MLXArray, mask: MLXArray, source: MLXArray
) -> MLXArray {
    let flattenedInput = inputTensor.flattened()
    let flattenedMask = mask.flattened().asArray(Bool.self)
    let flattenedSource = source.flattened()

    let targetIndices = flattenedMask.enumerated().compactMap { idx, value in
        value ? Int32(idx) : nil
    }
    guard !targetIndices.isEmpty else {
        return inputTensor
    }

    guard flattenedSource.dim(0) == targetIndices.count else {
        fatalError(
            "Masked scatter shape mismatch. source=\(flattenedSource.dim(0)) mask=\(targetIndices.count)"
        )
    }

    let result = flattenedInput
    result[MLXArray(targetIndices, [targetIndices.count])] = flattenedSource
    return result.reshaped(inputTensor.shape)
}

private func gemma4OneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< numClasses)
}

private func gemma4RotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape[x.shape.count - 1] / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

private func gemma4ApplyMultiDimensionalRoPE(
    _ inputs: MLXArray, positions: MLXArray, baseFrequency: Float
) -> MLXArray {
    let headDim = inputs.shape[inputs.ndim - 1]
    if positions.ndim == 2 {
        let half = headDim / 2
        let freqExponents =
            (2.0 / Float(headDim)) * MLXArray(0 ..< half).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let sinusoid = positions.asType(.float32).expandedDimensions(axis: -1) / timescale
        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)
        return inputs * cosValue + gemma4RotateHalf(inputs) * sinValue
    }

    let numDimensions = positions.shape[positions.ndim - 1]
    let channelsPerDimension = 2 * (headDim / (2 * numDimensions))
    let halfPerDimension = channelsPerDimension / 2

    var parts: [MLXArray] = []
    parts.reserveCapacity(numDimensions)

    for d in 0 ..< numDimensions {
        let start = d * channelsPerDimension
        let end = start + channelsPerDimension
        let part = inputs[.ellipsis, start ..< end]

        let freqExponents =
            (2.0 / Float(channelsPerDimension)) * MLXArray(0 ..< halfPerDimension).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let dimPositions = positions[.ellipsis, d ..< d + 1].asType(.float32)
        let sinusoid = dimPositions / timescale

        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)

        parts.append(part * cosValue + gemma4RotateHalf(part) * sinValue)
    }

    return concatenated(parts, axis: -1)
}

private func gemma4EnsureFusedSDPA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    let fusedDims = [64, 80, 128]
    let d = queries.dim(queries.ndim - 1)
    let target = fusedDims.first(where: { d <= $0 }) ?? d

    if target == d {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }

    let paddedQueries = MLX.padded(
        queries, widths: [0, 0, 0, .init((0, target - d))])
    let paddedKeys = MLX.padded(
        keys, widths: [0, 0, 0, .init((0, target - d))])
    let paddedValues = MLX.padded(
        values, widths: [0, 0, 0, .init((0, target - d))])

    return MLXFast.scaledDotProductAttention(
        queries: paddedQueries, keys: paddedKeys, values: paddedValues, scale: scale, mask: mask
    )[.ellipsis, ..<d]
}

private enum Gemma4SharedKVState {
    case regular(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: (MLXArray, MLXArray, MLXArray?),
        values: (MLXArray, MLXArray, MLXArray?),
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    )

    var sequenceLength: Int {
        switch self {
        case .regular(let keys, _):
            return keys.dim(2)
        case .quantized(let keys, _, _, _, _):
            return keys.0.dim(-2)
        }
    }
}

private func gemma4AdjustAttentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
    keyLength: Int
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mask {
    case .array(let maskArray):
        let maskLength = maskArray.dim(-1)
        guard maskLength > keyLength else {
            return mask
        }
        let start = maskLength - keyLength
        return .array(maskArray[.ellipsis, start...])
    case .arrays, .causal, .none:
        return mask
    }
}
// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let globalKVHeads: Int?
    public let headDim: Int
    public let globalHeadDim: Int
    public let vocabularySize: Int
    public let vocabularySizePerLayerInput: Int
    public let numKVSharedLayers: Int
    public let hiddenSizePerLayerInput: Int
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTraditional: Bool
    public let finalLogitSoftcapping: Float?
    public let useDoubleWideMLP: Bool
    public let enableMoEBlock: Bool
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    public let attentionKEqV: Bool
    public let layerTypes: [String]
    public let ropeParameters: [String: [String: StringOrNumber]]
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case vocabularySize = "vocab_size"
        case vocabularySizePerLayerInput = "vocab_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTraditional = "rope_traditional"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionKEqV = "attention_k_eq_v"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_text"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize) ?? 1536
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers) ?? 35
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize) ?? 6144
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads) ?? 8
        kvHeads = try c.decodeIfPresent(Int.self, forKey: CodingKeys.kvHeads) ?? 1
        globalKVHeads = try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalKVHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 256
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.globalHeadDim) ?? 512
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySize) ?? 262_144
        vocabularySizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.vocabularySizePerLayerInput)
            ?? vocabularySize
        numKVSharedLayers =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.numKVSharedLayers) ?? 20
        hiddenSizePerLayerInput =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSizePerLayerInput) ?? 256
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindow) ?? 512
        slidingWindowPattern =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.slidingWindowPattern) ?? 5
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxPositionEmbeddings) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        ropeTraditional =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.ropeTraditional) ?? false
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: CodingKeys.finalLogitSoftcapping) ?? 30.0
        useDoubleWideMLP =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useDoubleWideMLP) ?? true
        enableMoEBlock =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.enableMoEBlock) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.numExperts)
        topKExperts = try c.decodeIfPresent(Int.self, forKey: CodingKeys.topKExperts)
        moeIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: CodingKeys.moeIntermediateSize)
        attentionKEqV = try c.decodeIfPresent(Bool.self, forKey: CodingKeys.attentionKEqV) ?? false
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: CodingKeys.ropeParameters)
            ?? gemma4DefaultTextRopeParameters()
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: CodingKeys.layerTypes)
            ?? gemma4BuildLayerTypes(
                hiddenLayers: hiddenLayers, slidingWindowPattern: slidingWindowPattern)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings) ?? true
    }
}

public struct Gemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let rmsNormEps: Float
    public let defaultOutputLength: Int
    public let positionEmbeddingSize: Int
    public let poolingKernelSize: Int
    public let useClippedLinears: Bool
    public let standardize: Bool
    public let ropeParameters: [String: StringOrNumber]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case rmsNormEps = "rms_norm_eps"
        case defaultOutputLength = "default_output_length"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4_vision"
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenLayers) ?? 16
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.hiddenSize) ?? 768
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.intermediateSize) ?? 3072
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: CodingKeys.attentionHeads) ?? 12
        keyValueHeads =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.keyValueHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: CodingKeys.headDim) ?? 64
        patchSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize) ?? 16
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: CodingKeys.rmsNormEps) ?? 1e-6
        defaultOutputLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.defaultOutputLength) ?? 280
        positionEmbeddingSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.positionEmbeddingSize) ?? 10_240
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize) ?? 3
        useClippedLinears =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.useClippedLinears) ?? false
        standardize = try c.decodeIfPresent(Bool.self, forKey: CodingKeys.standardize) ?? false
        ropeParameters =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: CodingKeys.ropeParameters)
            ?? gemma4DefaultVisionRopeParameters()
    }
}

public struct Gemma4Configuration: Codable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4VisionConfiguration
    public let modelType: String
    public let quantization: BaseConfiguration.Quantization?
    public let imageTokenId: Int
    public let audioTokenId: Int?
    public let boiTokenId: Int
    public let eoiTokenId: Int?
    public let visionSoftTokensPerImage: Int
    public let tieWordEmbeddings: Bool

    private let _vocabularySize: Int?
    private let _hiddenSize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int { _vocabularySize ?? textConfiguration.vocabularySize }
    public var hiddenSize: Int { _hiddenSize ?? textConfiguration.hiddenSize }
    public var padTokenId: Int { _padTokenId ?? 0 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case quantization
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case tieWordEmbeddings = "tie_word_embeddings"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _padTokenId = "pad_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(
            Gemma4TextConfiguration.self, forKey: CodingKeys.textConfiguration)
        visionConfiguration = try c.decode(
            Gemma4VisionConfiguration.self, forKey: CodingKeys.visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: CodingKeys.modelType) ?? "gemma4"
        quantization = try c.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: CodingKeys.quantization)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.audioTokenId)
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId)
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.visionSoftTokensPerImage)
            ?? visionConfiguration.defaultOutputLength
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.tieWordEmbeddings)
            ?? textConfiguration.tieWordEmbeddings
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._vocabularySize)
        _hiddenSize = try c.decodeIfPresent(Int.self, forKey: CodingKeys._hiddenSize)
        _padTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys._padTokenId)
    }
}

// MARK: - Text

private final class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

private final class Gemma4RMSNormZeroShift: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

private final class Gemma4TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        let isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0
        let useDoubleWide = config.useDoubleWideMLP && isKVSharedLayer
        let hiddenDimensions = config.intermediateSize * (useDoubleWide ? 2 : 1)

        self._gateProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private final class Gemma4TextRouter: Module {
    let topKExperts: Int
    private let rootSize: Float

    @ModuleInfo(key: "norm") var norm: Gemma4RMSNormNoScale
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts, let topKExperts = config.topKExperts else {
            fatalError("Gemma4 MoE router requires numExperts and topKExperts")
        }

        self.topKExperts = topKExperts
        self.rootSize = pow(Float(config.hiddenSize), -0.5)

        self._norm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)
        self._proj.wrappedValue = Linear(config.hiddenSize, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([config.hiddenSize])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var x = norm(x)
        x = x * MLXArray(rootSize, dtype: x.dtype)
        x = x * scale.asType(x.dtype)

        let expertScores = proj(x)
        let routerProbabilities = MLX.softmax(expertScores, axis: -1, precise: true)

        let topKIndices = MLX.argPartition(-expertScores, kth: topKExperts - 1, axis: -1)[
            .ellipsis, ..<topKExperts,
        ]
        var topKWeights = MLX.takeAlong(routerProbabilities, topKIndices, axis: -1)
        topKWeights = topKWeights / MLX.sum(topKWeights, axis: -1, keepDims: true)
        topKWeights = topKWeights * perExpertScale[topKIndices].asType(topKWeights.dtype)
        return (topKIndices, topKWeights)
    }
}

private final class Gemma4TextExperts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(config: Gemma4TextConfiguration) {
        guard let numExperts = config.numExperts,
            let moeIntermediateSize = config.moeIntermediateSize
        else {
            fatalError("Gemma4 MoE experts require numExperts and moeIntermediateSize")
        }

        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = x.dim(2)
        let topK = topKIndices.dim(-1)

        let expertOutput = switchGLU(
            x.reshaped(batch * length, hidden),
            topKIndices.reshaped(batch * length, topK)
        )
        let weights = topKWeights.reshaped(batch * length, topK, 1).asType(expertOutput.dtype)
        return (expertOutput * weights).sum(axis: -2).reshaped(batch, length, hidden)
    }
}

private final class Gemma4ScaledLinear: Module, UnaryLayer {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let scalar: Float

    init(inFeatures: Int, outFeatures: Int, scalar: Float) {
        self.scalar = scalar
        self._weight.wrappedValue = MLXArray.zeros([outFeatures, inFeatures])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x.matmul(weight.transposed())) * scalar
    }
}

private final class Gemma4TextAttention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let headDim: Int
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float
    let isKVSharedLayer: Bool
    let useKEqV: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "v_norm") var vNorm: Gemma4RMSNormNoScale
    @ModuleInfo var rope: OffsetLayer

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.headDim =
            layerType == "full_attention" && config.globalHeadDim > 0
            ? config.globalHeadDim : config.headDim
        self.numHeads = config.attentionHeads
        self.useKEqV = config.attentionKEqV && !isSliding
        self.numKVHeads =
            useKEqV ? (config.globalKVHeads ?? config.kvHeads) : config.kvHeads
        self.scale = 1.0

        let firstKVSharedLayer = config.hiddenLayers - config.numKVSharedLayers
        self.isKVSharedLayer = layerIdx >= firstKVSharedLayer && firstKVSharedLayer > 0

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        if !useKEqV {
            self._vProj.wrappedValue = Linear(
                config.hiddenSize, numKVHeads * headDim, bias: false)
        }
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)

        let ropeKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeConfig = config.ropeParameters[ropeKey]
        let ropeTheta = ropeConfig?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000 : 1_000_000)
        self._rope.wrappedValue = initializeRope(
            dims: headDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        queries = qNorm(queries)

        let currentOffset: Int
        let kvState: Gemma4SharedKVState?

        if let sharedKV {
            currentOffset = offset ?? 0
            kvState = sharedKV
        } else {
            currentOffset = cache?.offset ?? 0
            var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
            var values =
                if useKEqV {
                    keys
                } else {
                    vProj!(x).reshaped(batch, length, numKVHeads, headDim)
                }
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            values = vNorm(values).transposed(0, 2, 1, 3)
            keys = rope(keys, offset: currentOffset)
            if let quantizedCache = cache as? QuantizedKVCacheProtocol {
                let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(
                    keys: keys, values: values)
                kvState = .quantized(
                    keys: quantizedKeys,
                    values: quantizedValues,
                    groupSize: quantizedCache.groupSize,
                    bits: quantizedCache.bits,
                    mode: quantizedCache.mode
                )
            } else {
                if let cache {
                    (keys, values) = cache.update(keys: keys, values: values)
                }
                kvState = .regular(keys: keys, values: values)
            }
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: currentOffset)

        guard let kvState else {
            fatalError("Gemma4 attention expected a KV state")
        }
        let localMask = gemma4AdjustAttentionMask(mask, keyLength: kvState.sequenceLength)

        let output: MLXArray =
            switch kvState {
            case .regular(let keys, let values):
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: localMask
                )
            case .quantized(let keys, let values, let groupSize, let bits, let mode):
                quantizedScaledDotProductAttention(
                    queries: queries,
                    quantizedKeys: keys,
                    quantizedValues: values,
                    scale: scale,
                    mask: localMask,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
            }

        return (
            oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)),
            kvState,
            currentOffset
        )
    }
}

private final class Gemma4TextDecoderLayer: Module {
    let layerType: String
    let enableMoE: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4TextAttention
    @ModuleInfo var mlp: Gemma4TextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "router") var router: Gemma4TextRouter?
    @ModuleInfo(key: "experts") var experts: Gemma4TextExperts?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2:
        Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNormZeroShift?
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(config: Gemma4TextConfiguration, layerIdx: Int) {
        self.layerType = config.layerTypes[layerIdx]
        self.enableMoE = config.enableMoEBlock
        self._selfAttention.wrappedValue = Gemma4TextAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4TextMLP(config: config, layerIdx: layerIdx)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4TextRouter(config: config)
            self._experts.wrappedValue = Gemma4TextExperts(config: config)
            self._postFeedforwardLayerNorm1.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        if config.hiddenSizePerLayerInput > 0 {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjection.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: Gemma4SharedKVState? = nil,
        offset: Int? = nil
    ) -> (MLXArray, Gemma4SharedKVState?, Int) {
        var residual = x
        var h = inputLayerNorm(x)
        let (attentionOutput, kvState, attentionOffset) = selfAttention(
            h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = attentionOutput
        h = postAttentionLayerNorm(h)
        h = residual + h

        residual = h
        if enableMoE,
            let router,
            let experts,
            let postFeedforwardLayerNorm1,
            let postFeedforwardLayerNorm2,
            let preFeedforwardLayerNorm2
        {
            var dense = preFeedforwardLayerNorm(h)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let (topKIndices, topKWeights) = router(h)
            var sparse = preFeedforwardLayerNorm2(h)
            sparse = experts(sparse, topKIndices: topKIndices, topKWeights: topKWeights)
            sparse = postFeedforwardLayerNorm2(sparse)

            h = dense + sparse
        } else {
            h = preFeedforwardLayerNorm(h)
            h = mlp(h)
        }
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        if let perLayerInputGate, let perLayerProjection, let postPerLayerInputNorm,
            let perLayerInput
        {
            residual = h
            var gated = perLayerInputGate(h)
            gated = geluApproximate(gated)
            gated = gated * perLayerInput
            gated = perLayerProjection(gated)
            gated = postPerLayerInputNorm(gated)
            h = residual + gated
        }

        return (h * layerScalar, kvState, attentionOffset)
    }
}

private final class Gemma4TextBackbone: Module {
    let config: Gemma4TextConfiguration
    let firstKVSharedLayerIdx: Int
    let layerIdxToCacheIdx: [Int]
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int
    let embedScale: Float
    let embedTokensPerLayerScale: Float
    private let _perLayerInputScale: MLXArray

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4TextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Gemma4ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm:
        Gemma4RMSNormZeroShift?

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.firstKVSharedLayerIdx = config.hiddenLayers - config.numKVSharedLayers
        self.embedScale = pow(Float(config.hiddenSize), 0.5)
        self.embedTokensPerLayerScale = pow(Float(max(config.hiddenSizePerLayerInput, 1)), 0.5)
        self._perLayerInputScale = rsqrt(MLXArray(2.0))

        let concreteLayers = Array(config.layerTypes.prefix(firstKVSharedLayerIdx))
        let sharedFullIdx = concreteLayers.lastIndex(of: "full_attention") ?? 0
        let sharedSlidingIdx = concreteLayers.lastIndex(of: "sliding_attention") ?? 0

        var cacheMap: [Int] = []
        cacheMap.reserveCapacity(config.hiddenLayers)
        for (idx, layerType) in config.layerTypes.enumerated() {
            if idx < firstKVSharedLayerIdx {
                cacheMap.append(idx)
            } else {
                cacheMap.append(layerType == "full_attention" ? sharedFullIdx : sharedSlidingIdx)
            }
        }
        layerIdxToCacheIdx = cacheMap
        firstFullCacheIdx = concreteLayers.firstIndex(of: "full_attention") ?? 0
        firstSlidingCacheIdx = concreteLayers.firstIndex(of: "sliding_attention") ?? 0

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Gemma4TextDecoderLayer(config: config, layerIdx: $0)
        }
        self._norm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.hiddenSizePerLayerInput > 0 {
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabularySizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
            )
            self._perLayerModelProjection.wrappedValue = Gemma4ScaledLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.hiddenLayers * config.hiddenSizePerLayerInput,
                scalar: pow(Float(config.hiddenSize), -0.5)
            )
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        super.init()
    }

    func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embedTokensPerLayer else {
            fatalError("Per-layer inputs requested for a model without embed_tokens_per_layer")
        }
        let validMask =
            logicalAnd(
                inputIds .>= 0, inputIds .< config.vocabularySizePerLayerInput)
        let tokens = MLX.where(validMask, inputIds, MLXArray.zeros(like: inputIds))
        var result = embedTokensPerLayer(tokens)
        result = (result * MLXArray(embedTokensPerLayerScale, dtype: .float32)).asType(result.dtype)
        return result.reshaped(
            Array(inputIds.shape) + [config.hiddenLayers, config.hiddenSizePerLayerInput]
        )
    }

    func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray, perLayerInputs: MLXArray?
    ) -> MLXArray? {
        guard let perLayerModelProjection, let perLayerProjectionNorm else {
            return nil
        }

        var perLayerProjection = perLayerModelProjection(inputsEmbeds)
        perLayerProjection = perLayerProjection.reshaped(
            Array(inputsEmbeds.shape.dropLast()) + [
                config.hiddenLayers, config.hiddenSizePerLayerInput,
            ]
        )
        perLayerProjection = perLayerProjectionNorm(perLayerProjection)

        guard let perLayerInputs else {
            return perLayerProjection
        }

        return (perLayerProjection + perLayerInputs)
            * _perLayerInputScale.asType(inputsEmbeds.dtype)
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
        let h0: MLXArray
        if let inputsEmbeds {
            h0 = inputsEmbeds
        } else if let inputs {
            let embeddings = embedTokens(inputs)
            h0 = (embeddings * MLXArray(embedScale, dtype: .float32)).asType(embeddings.dtype)
        } else {
            fatalError("Either inputs or inputsEmbeds must be provided")
        }

        let processedPerLayerInputs: MLXArray?
        if config.hiddenSizePerLayerInput > 0 {
            if let perLayerInputs {
                processedPerLayerInputs = perLayerInputs
            } else if let inputs {
                processedPerLayerInputs = getPerLayerInputs(inputs)
            } else {
                processedPerLayerInputs = nil
            }
        } else {
            processedPerLayerInputs = nil
        }
        let finalPerLayerInputs = projectPerLayerInputs(h0, perLayerInputs: processedPerLayerInputs)

        let hasExplicitCache = cache != nil
        let localCache =
            cache ?? Array(repeating: nil as KVCache?, count: max(firstKVSharedLayerIdx, 1))
        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let mask {
            fullMask = mask
            slidingMask = mask
        } else {
            fullMask = createAttentionMask(
                h: h0,
                cache: firstFullCacheIdx < localCache.count ? localCache[firstFullCacheIdx] : nil)
            slidingMask = createAttentionMask(
                h: h0,
                cache: firstSlidingCacheIdx < localCache.count
                    ? localCache[firstSlidingCacheIdx] : nil,
                windowSize: config.slidingWindow
            )
        }

        var h = h0
        var intermediates = [(kv: Gemma4SharedKVState?, offset: Int?)](
            repeating: (nil, nil), count: config.hiddenLayers)
        for (idx, layer) in layers.enumerated() {
            let sourceIdx = layerIdxToCacheIdx[idx]
            let layerCache: KVCache? =
                if idx < firstKVSharedLayerIdx, sourceIdx < localCache.count {
                    localCache[sourceIdx]
                } else {
                    nil
                }
            let layerMask =
                if layer.layerType == "full_attention" {
                    fullMask
                } else {
                    slidingMask
                }
            let layerInput: MLXArray? =
                if let finalPerLayerInputs {
                    finalPerLayerInputs[0..., 0..., idx, 0...]
                } else {
                    nil
                }
            let (output, kvState, attentionOffset) = layer(
                h,
                mask: layerMask,
                cache: layerCache,
                perLayerInput: layerInput,
                sharedKV: hasExplicitCache && idx >= firstKVSharedLayerIdx
                    ? intermediates[sourceIdx].kv : nil,
                offset: hasExplicitCache && idx >= firstKVSharedLayerIdx
                    ? intermediates[sourceIdx].offset : nil
            )
            h = output
            intermediates[idx] = (kvState, attentionOffset)
        }
        return norm(h)
    }
}

private final class Gemma4TextLanguageModel: Module, KVCacheDimensionProvider {
    let config: Gemma4TextConfiguration
    let finalLogitSoftcapping: Float?

    @ModuleInfo(key: "model") var model: Gemma4TextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    var kvHeads: [Int] {
        (0 ..< config.hiddenLayers).map { idx in
            let layerType = config.layerTypes[idx]
            if config.attentionKEqV && layerType == "full_attention" {
                return config.globalKVHeads ?? config.kvHeads
            } else {
                return config.kvHeads
            }
        }
    }

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self._model.wrappedValue = Gemma4TextBackbone(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        let slidingWindow = config.slidingWindow > 0 ? config.slidingWindow : 4096
        return config.layerTypes.prefix(config.hiddenLayers - config.numKVSharedLayers).map {
            layerType in
            if layerType == "full_attention" {
                StandardKVCache()
            } else {
                RotatingKVCache(maxSize: slidingWindow, keep: 0)
            }
        }
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil,
        cache: [KVCache]? = nil,
        inputsEmbeds: MLXArray? = nil,
        perLayerInputs: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> LMOutput {
        let output = model(
            inputs, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache?.map { $0 as KVCache? },
            perLayerInputs: perLayerInputs
        )
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(output)
        } else {
            logits = model.embedTokens.asLinear(output)
        }
        if let finalLogitSoftcapping, finalLogitSoftcapping > 0 {
            let scale = MLXArray(finalLogitSoftcapping)
            return LMOutput(logits: tanh(logits / scale) * scale)
        }
        return LMOutput(logits: logits)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count + 1)

        for (key, value) in weights {
            if key.contains("rotary_emb") {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            }
            if newKey.hasPrefix("language_model."),
                !newKey.hasPrefix("language_model.model."),
                !newKey.hasPrefix("language_model.lm_head.")
            {
                let rest = String(newKey.dropFirst("language_model.".count))
                newKey = "language_model.model.\(rest)"
            }

            if newKey.hasSuffix(".experts.down_proj") {
                newKey = newKey.replacingOccurrences(
                    of: ".experts.down_proj",
                    with: ".experts.switch_glu.down_proj.weight"
                )
            }

            if newKey.hasSuffix(".experts.gate_up_proj") {
                let mid = value.dim(-2) / 2
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.gate_proj.weight"
                    )
                ] = value[.ellipsis, ..<mid, 0...]
                sanitized[
                    newKey.replacingOccurrences(
                        of: ".experts.gate_up_proj",
                        with: ".experts.switch_glu.up_proj.weight"
                    )
                ] = value[.ellipsis, mid..., 0...]
                continue
            }

            sanitized[newKey] = value
        }

        if config.tieWordEmbeddings {
            sanitized = sanitized.filter { key, _ in
                !key.hasPrefix("language_model.lm_head.")
            }
        } else if sanitized["language_model.lm_head.weight"] == nil,
            let embedWeight = sanitized["language_model.model.embed_tokens.weight"]
        {
            sanitized["language_model.lm_head.weight"] = embedWeight
        }

        return sanitized
    }
}

// MARK: - Vision

private final class Gemma4ClippableLinear: Module, UnaryLayer {
    let useClipping: Bool

    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "input_min") var inputMin: MLXArray?
    @ModuleInfo(key: "input_max") var inputMax: MLXArray?
    @ModuleInfo(key: "output_min") var outputMin: MLXArray?
    @ModuleInfo(key: "output_max") var outputMax: MLXArray?

    init(inFeatures: Int, outFeatures: Int, bias: Bool = false, useClipping: Bool) {
        self.useClipping = useClipping
        self._linear.wrappedValue = Linear(inFeatures, outFeatures, bias: bias)
        if useClipping {
            self._inputMin.wrappedValue = MLXArray(-Float.infinity)
            self._inputMax.wrappedValue = MLXArray(Float.infinity)
            self._outputMin.wrappedValue = MLXArray(-Float.infinity)
            self._outputMax.wrappedValue = MLXArray(Float.infinity)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let clippedInput =
            if let inputMin, let inputMax {
                clip(x, min: inputMin, max: inputMax)
            } else {
                x
            }
        let projected = linear(clippedInput)
        if let outputMin, let outputMax {
            return clip(projected, min: outputMin, max: outputMax)
        }
        return projected
    }
}

private final class Gemma4VisionRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat.square(), axis: -1, keepDims: true)
        let normalized = xFloat * rsqrt(variance + eps)
        return (normalized * weight.asType(.float32)).asType(x.dtype)
    }
}

private final class Gemma4VisionRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat.square(), axis: -1, keepDims: true)
        return (xFloat * rsqrt(variance + eps)).asType(x.dtype)
    }
}

private final class Gemma4VisionAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let hiddenSize: Int
    let ropeBaseFrequency: Float

    @ModuleInfo(key: "q_proj") var qProj: Gemma4ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: Gemma4ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: Gemma4ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: Gemma4ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma4VisionRMSNorm
    @ModuleInfo(key: "_v_norm") var vNorm: Gemma4VisionRMSNormNoScale

    init(config: Gemma4VisionConfiguration) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.keyValueHeads
        self.headDim = config.headDim
        self.hiddenSize = config.hiddenSize
        self.ropeBaseFrequency = config.ropeParameters["rope_theta"]?.asFloat() ?? 100.0

        self._qProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._kProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numKVHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._vProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numKVHeads * headDim,
            useClipping: config.useClippedLinears
        )
        self._oProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: numHeads * headDim,
            outFeatures: hiddenSize,
            useClipping: config.useClippedLinears
        )
        self._qNorm.wrappedValue = Gemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = Gemma4VisionRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._vNorm.wrappedValue = Gemma4VisionRMSNormNoScale(eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
        var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
        var values = vProj(x).reshaped(batch, length, numKVHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)
        values = vNorm(values)

        queries = gemma4ApplyMultiDimensionalRoPE(
            queries, positions: positions, baseFrequency: ropeBaseFrequency)
        keys = gemma4ApplyMultiDimensionalRoPE(
            keys, positions: positions, baseFrequency: ropeBaseFrequency)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let mask {
                .array(mask)
            } else {
                .none
            }
        let output = gemma4EnsureFusedSDPA(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(output)
    }
}

private final class Gemma4VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: Gemma4ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: Gemma4ClippableLinear

    init(config: Gemma4VisionConfiguration) {
        self._gateProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useClipping: config.useClippedLinears
        )
        self._upProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.intermediateSize,
            useClipping: config.useClippedLinears
        )
        self._downProj.wrappedValue = Gemma4ClippableLinear(
            inFeatures: config.intermediateSize,
            outFeatures: config.hiddenSize,
            useClipping: config.useClippedLinears
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private final class Gemma4VisionTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4VisionAttention
    @ModuleInfo var mlp: Gemma4VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNormZeroShift
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        Gemma4RMSNormZeroShift

    init(config: Gemma4VisionConfiguration) {
        self._selfAttention.wrappedValue = Gemma4VisionAttention(config: config)
        self._mlp.wrappedValue = Gemma4VisionMLP(config: config)
        self._inputLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNormZeroShift(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let normed = inputLayerNorm(x)
        let attentionOutput = selfAttention(normed, positions: positions, mask: mask)
        let h = x + postAttentionLayerNorm(attentionOutput)
        let ff = mlp(preFeedforwardLayerNorm(h))
        return h + postFeedforwardLayerNorm(ff)
    }
}

private final class Gemma4VisionPatchEmbedder: Module {
    let patchSize: Int
    let hiddenSize: Int
    let positionEmbeddingSize: Int

    @ModuleInfo(key: "input_proj") var inputProjection: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    init(config: Gemma4VisionConfiguration) {
        self.patchSize = config.patchSize
        self.hiddenSize = config.hiddenSize
        self.positionEmbeddingSize = config.positionEmbeddingSize
        self._inputProjection.wrappedValue = Linear(
            3 * patchSize * patchSize, hiddenSize, bias: false)
        self._positionEmbeddingTable.wrappedValue = MLXArray.ones([
            2, positionEmbeddingSize, hiddenSize,
        ])
        super.init()
    }

    private func patchify(_ pixelValues: MLXArray) -> MLXArray {
        let (batch, channels, height, width) = (
            pixelValues.dim(0), pixelValues.dim(1), pixelValues.dim(2), pixelValues.dim(3)
        )
        let patchesH = height / patchSize
        let patchesW = width / patchSize

        var patches = pixelValues.reshaped(
            batch, channels, patchesH, patchSize, patchesW, patchSize)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(batch, patchesH * patchesW, channels * patchSize * patchSize)
        patches = 2 * (patches - 0.5)
        return inputProjection(patches.asType(inputProjection.weight.dtype))
    }

    func callAsFunction(
        _ pixelValues: MLXArray, patchPositions: MLXArray
    ) -> MLXArray {
        let hiddenStates = patchify(pixelValues)
        let batch = patchPositions.dim(0)
        let seqLen = patchPositions.dim(1)

        let xIndices = patchPositions[0..., 0..., 0].flattened().asType(.int32)
        let yIndices = patchPositions[0..., 0..., 1].flattened().asType(.int32)
        let xEmbeddings = take(positionEmbeddingTable[0], xIndices, axis: 0)
            .reshaped(batch, seqLen, hiddenSize)
        let yEmbeddings = take(positionEmbeddingTable[1], yIndices, axis: 0)
            .reshaped(batch, seqLen, hiddenSize)
        return hiddenStates + xEmbeddings + yEmbeddings
    }
}

private final class Gemma4VisionPooler: Module {
    let hiddenSize: Int
    let rootHiddenSize: Float

    init(config: Gemma4VisionConfiguration) {
        self.hiddenSize = config.hiddenSize
        self.rootHiddenSize = pow(Float(config.hiddenSize), 0.5)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        patchPositions: MLXArray,
        outputLength: Int
    ) -> MLXArray {
        let scale = MLXArray(rootHiddenSize, dtype: hiddenStates.dtype)
        let numPatches = hiddenStates.dim(1)
        let length = max(outputLength, 1)
        if numPatches <= length {
            return hiddenStates * scale
        }

        // All batch rows share one position grid, so a single [l, L] weight
        // matrix pools every row. The processor's resize keeps both sides
        // divisible by kernel * patchSize, so the pooled grid covers the
        // image exactly.
        let positions = patchPositions[0]
        let maxX = Int(positions[0..., 0].max().item(Int32.self)) + 1
        let kernel = max(Int(sqrt(Double(numPatches / length))), 1)
        let divisor = kernel * kernel

        let kernelIndices = floor(positions.asType(.float32) / Float(kernel)).asType(.int32)
        let flatKernel =
            kernelIndices[0..., 0] + MLXArray(Int32(max(maxX / kernel, 1)))
            * kernelIndices[0..., 1]
        let weights =
            gemma4OneHot(flatKernel, numClasses: length).asType(.float32)
            / Float(divisor)
        let output = einsum("lL,bld->bLd", weights, hiddenStates)
            .asType(hiddenStates.dtype)
        return output * scale
    }
}

private final class Gemma4VisionTransformerModel: Module {
    @ModuleInfo(key: "layers") var layers: [Gemma4VisionTransformerBlock]

    init(config: Gemma4VisionConfiguration) {
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma4VisionTransformerBlock(config: config)
        }
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray
    {
        var h = hiddenStates
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}

private final class Gemma4VisionModel: Module {
    let config: Gemma4VisionConfiguration
    let patchSize: Int
    let poolingKernelSize: Int

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: Gemma4VisionPatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: Gemma4VisionTransformerModel
    @ModuleInfo(key: "pooler") var pooler: Gemma4VisionPooler
    @ModuleInfo(key: "std_bias") var standardizationBias: MLXArray?
    @ModuleInfo(key: "std_scale") var standardizationScale: MLXArray?

    init(config: Gemma4VisionConfiguration) {
        self.config = config
        self.patchSize = config.patchSize
        self.poolingKernelSize = config.poolingKernelSize
        self._patchEmbedder.wrappedValue = Gemma4VisionPatchEmbedder(config: config)
        self._encoder.wrappedValue = Gemma4VisionTransformerModel(config: config)
        self._pooler.wrappedValue = Gemma4VisionPooler(config: config)
        if config.standardize {
            self._standardizationBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            self._standardizationScale.wrappedValue = MLXArray.ones([config.hiddenSize])
        }
        super.init()
    }

    private func patchPositions(batch: Int, patchesH: Int, patchesW: Int) -> MLXArray {
        var values = [Int32]()
        values.reserveCapacity(patchesH * patchesW * 2)
        for y in 0 ..< patchesH {
            for x in 0 ..< patchesW {
                values.append(Int32(x))
                values.append(Int32(y))
            }
        }
        let positions = MLXArray(values, [1, patchesH * patchesW, 2])
        return batch == 1
            ? positions
            : broadcast(positions, to: [batch, patchesH * patchesW, 2])
    }

    /// Encodes a batch of same-sized images. Every patch is real (callers
    /// slice padded canvases down to each image's true size first), so
    /// attention is dense and the pooled output length falls out of the
    /// patch grid: numPatches / poolingKernelSize².
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let pixels =
            if pixelValues.ndim == 3 {
                expandedDimensions(pixelValues, axis: 0)
            } else {
                pixelValues
            }
        let batch = pixels.dim(0)
        let patchesH = pixels.dim(2) / patchSize
        let patchesW = pixels.dim(3) / patchSize
        let numPatches = patchesH * patchesW
        let outputLength = max(numPatches / (poolingKernelSize * poolingKernelSize), 1)

        let patchPositions = patchPositions(batch: batch, patchesH: patchesH, patchesW: patchesW)
        var hiddenStates = patchEmbedder(pixels, patchPositions: patchPositions)
        hiddenStates = encoder(hiddenStates, positions: patchPositions, mask: nil)
        hiddenStates = pooler(
            hiddenStates, patchPositions: patchPositions, outputLength: outputLength)

        if let standardizationBias, let standardizationScale {
            hiddenStates = (hiddenStates - standardizationBias) * standardizationScale
        }
        return hiddenStates
    }
}

private final class Gemma4MultimodalEmbedder: Module, UnaryLayer {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm:
        Gemma4RMSNormNoScale

    init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
        self._embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
        self._embeddingPreProjectionNorm.wrappedValue = Gemma4RMSNormNoScale(eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        embeddingProjection(embeddingPreProjectionNorm(x))
    }
}

// MARK: - Model

public final class Gemma4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4TextLanguageModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder

    public let config: Gemma4Configuration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        self._visionTower.wrappedValue = Gemma4VisionModel(config: config.visionConfiguration)
        self._languageModel.wrappedValue = Gemma4TextLanguageModel(config.textConfiguration)
        self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
            embeddingDim: config.visionConfiguration.hiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            eps: config.visionConfiguration.rmsNormEps
        )
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        image: LMInput.ProcessedImage? = nil
    ) throws -> (MLXArray, MLXArray?) {
        var inputsEmbeds = languageModel.model.embedTokens(inputIds)
        inputsEmbeds =
            (inputsEmbeds
            * MLXArray(pow(Float(config.textConfiguration.hiddenSize), 0.5), dtype: .float32))
            .asType(inputsEmbeds.dtype)

        var perLayerInputs: MLXArray? = nil
        if config.textConfiguration.hiddenSizePerLayerInput > 0 {
            let imageMask = inputIds .== config.imageTokenId
            let audioMask =
                if let audioTokenId = config.audioTokenId {
                    inputIds .== audioTokenId
                } else {
                    MLXArray.zeros(like: imageMask)
                }
            let textMask = logicalNot(logicalOr(imageMask, audioMask))
            let perLayerTokens = MLX.where(textMask, inputIds, MLXArray.zeros(like: inputIds))
            perLayerInputs = languageModel.model.getPerLayerInputs(perLayerTokens)
        }

        guard let image else {
            return (inputsEmbeds, perLayerInputs)
        }

        // Images keep their own aspect-preserving sizes: the processor
        // zero-pads them onto a shared canvas and records each real size in
        // frames. Slice each image back out, run the tower on it alone, and
        // concatenate the pooled tokens in placeholder order.
        let pixels =
            if image.pixels.ndim == 3 {
                expandedDimensions(image.pixels, axis: 0)
            } else {
                image.pixels
            }
        let frames =
            image.frames
            ?? Array(repeating: THW(1, pixels.dim(2), pixels.dim(3)), count: pixels.dim(0))
        var perImageFeatures: [MLXArray] = []
        for (index, frame) in frames.enumerated() {
            let imagePixels = pixels[index ..< index + 1, 0..., ..<frame.h, ..<frame.w]
            perImageFeatures.append(visionTower(imagePixels))
        }
        var imageFeatures =
            perImageFeatures.count == 1
            ? perImageFeatures[0]
            : concatenated(perImageFeatures, axis: 1)
        imageFeatures = embedVision(imageFeatures)
        imageFeatures = imageFeatures.asType(inputsEmbeds.dtype)

        let imageMask = inputIds .== config.imageTokenId
        let expectedImageTokens = imageMask.asType(.int32).sum().item(Int.self)

        if expectedImageTokens != imageFeatures.dim(1) {
            throw Gemma4Error.imageTokenCountMismatch(
                expectedVisionTokens: imageFeatures.dim(1), actualPromptTokens: expectedImageTokens)
        }

        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        imageMaskExpanded = broadcast(imageMaskExpanded, to: inputsEmbeds.shape)
        inputsEmbeds = gemma4MaskedScatter(
            inputTensor: inputsEmbeds,
            mask: imageMaskExpanded,
            source: imageFeatures
        )

        return (inputsEmbeds, perLayerInputs)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let convertedCache = cache.map { $0 }
        if let image = input.image {
            let (inputsEmbeds, perLayerInputs) = try getInputEmbeddings(
                inputIds: input.text.tokens, image: image)
            let result = languageModel(
                nil,
                cache: convertedCache,
                inputsEmbeds: inputsEmbeds,
                perLayerInputs: perLayerInputs
            )
            return .logits(result)
        } else {
            let result = languageModel(input.text.tokens, cache: convertedCache)
            return .logits(result)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let logits = languageModel(inputs, cache: cache?.map { $0 })
        return logits.logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = languageModel.sanitize(weights: weights)

        // This port currently supports text + vision only.
        sanitized = sanitized.filter { key, _ in
            !key.contains("audio_tower") && !key.contains("embed_audio")
        }

        if !config.visionConfiguration.useClippedLinears {
            sanitized = sanitized.filter { key, _ in
                !key.contains("input_min")
                    && !key.contains("input_max")
                    && !key.contains("output_min")
                    && !key.contains("output_max")
            }
        }

        return sanitized
    }
}

// MARK: - Processor

public struct Gemma4MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        if message.role == .system {
            [
                "role": message.role.rawValue,
                "content": message.content,
            ]
        } else {
            [
                "role": message.role.rawValue,
                "content": message.images.map { _ in
                    ["type": "image"]
                }
                    + message.videos.map { _ in
                        ["type": "video"]
                    }
                    + [
                        ["type": "text", "text": message.content]
                    ],
            ]
        }
    }
}

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        let processedImage = MediaProcessing.apply(image, processing: processing)
        let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
        let targetSize = config.aspectPreservingTargetSize(for: srgbImage.extent.size)
        let resizedImage =
            srgbImage.extent.size == targetSize
            ? srgbImage
            : MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
        let finalImage =
            if config.doNormalize {
                MediaProcessing.normalize(
                    resizedImage, mean: config.imageMeanTuple, std: config.imageStdTuple)
            } else {
                resizedImage
            }
        let pixelValues = MediaProcessing.asMLXArray(finalImage)

        return (pixelValues, THW(1, Int(targetSize.height), Int(targetSize.width)))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(image: $0.asCIImage(), processing: input.processing)
            }
            let frames = imagePixelsAndFrames.map { $0.1 }

            // Each image keeps its own aspect-preserving size. ProcessedImage
            // carries one array, so zero-pad every image onto the largest
            // canvas in the request; the model slices the real regions back
            // out using frames.
            let maxHeight = frames.map(\.h).max() ?? 0
            let maxWidth = frames.map(\.w).max() ?? 0
            let paddedPixels = imagePixelsAndFrames.map { pixels, frame in
                frame.h == maxHeight && frame.w == maxWidth
                    ? pixels
                    : MLX.padded(
                        pixels,
                        widths: [
                            0, 0, .init((0, maxHeight - frame.h)), .init((0, maxWidth - frame.w)),
                        ])
            }
            processedImage = LMInput.ProcessedImage(
                pixels: concatenated(paddedPixels), frames: frames)

            // Expand the i-th image placeholder to that image's soft token
            // count: numPatches / poolingKernelSize².
            let softTokenCounts = frames.map { config.softTokenCount(height: $0.h, width: $0.w) }
            var expandedTokens: [Int] = []
            var imageIndex = 0
            for token in promptTokens {
                if token == config.imageTokenId {
                    guard imageIndex < softTokenCounts.count else {
                        throw Gemma4Error.imagePlaceholderMismatch(
                            images: softTokenCounts.count, placeholders: imageIndex + 1)
                    }
                    expandedTokens.append(config.boiTokenId)
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: config.imageTokenId,
                            count: softTokenCounts[imageIndex]))
                    if let eoiTokenId = config.eoiTokenId {
                        expandedTokens.append(eoiTokenId)
                    }
                    imageIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(text: .init(tokens: promptArray, mask: mask), image: processedImage)
    }
}

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let maxSoftTokens: Int
    public let patchSize: Int
    public let poolingKernelSize: Int

    public let imageTokenId: Int
    public let boiTokenId: Int
    public let eoiTokenId: Int?

    /// Image keys nested under `image_processor` in processor_config.json.
    /// Repos that ship a flat preprocessor_config.json put the same keys at
    /// the top level, which wins when both are present.
    private struct ImageProcessorConfiguration: Codable {
        let doNormalize: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let imageSeqLength: Int?
        let maxSoftTokens: Int?
        let patchSize: Int?
        let poolingKernelSize: Int?

        enum CodingKeys: String, CodingKey {
            case doNormalize = "do_normalize"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case imageSeqLength = "image_seq_length"
            case maxSoftTokens = "max_soft_tokens"
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
        }
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case maxSoftTokens = "max_soft_tokens"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case imageProcessor = "image_processor"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(
            ImageProcessorConfiguration.self, forKey: CodingKeys.imageProcessor)
        processorClass = try c.decode(String.self, forKey: CodingKeys.processorClass)
        doNormalize =
            try c.decodeIfPresent(Bool.self, forKey: CodingKeys.doNormalize)
            ?? nested?.doNormalize ?? false
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageMean)
            ?? nested?.imageMean ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: CodingKeys.imageStd)
            ?? nested?.imageStd ?? [0.5, 0.5, 0.5]
        imageSeqLength =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageSeqLength)
            ?? nested?.imageSeqLength ?? 280
        maxSoftTokens =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.maxSoftTokens)
            ?? nested?.maxSoftTokens ?? 280
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.patchSize)
            ?? nested?.patchSize ?? 16
        poolingKernelSize =
            try c.decodeIfPresent(Int.self, forKey: CodingKeys.poolingKernelSize)
            ?? nested?.poolingKernelSize ?? 3
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.imageTokenId) ?? 258_880
        boiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.boiTokenId) ?? 255_999
        eoiTokenId = try c.decodeIfPresent(Int.self, forKey: CodingKeys.eoiTokenId) ?? 258_882
    }

    public func encode(to encoder: any Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(processorClass, forKey: CodingKeys.processorClass)
        try c.encode(doNormalize, forKey: CodingKeys.doNormalize)
        try c.encode(imageMean, forKey: CodingKeys.imageMean)
        try c.encode(imageStd, forKey: CodingKeys.imageStd)
        try c.encode(imageSeqLength, forKey: CodingKeys.imageSeqLength)
        try c.encode(maxSoftTokens, forKey: CodingKeys.maxSoftTokens)
        try c.encode(patchSize, forKey: CodingKeys.patchSize)
        try c.encode(poolingKernelSize, forKey: CodingKeys.poolingKernelSize)
        try c.encode(imageTokenId, forKey: CodingKeys.imageTokenId)
        try c.encode(boiTokenId, forKey: CodingKeys.boiTokenId)
        try c.encodeIfPresent(eoiTokenId, forKey: CodingKeys.eoiTokenId)
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    /// Soft tokens the vision tower produces for an image of the given
    /// (already resized) pixel dimensions.
    public func softTokenCount(height: Int, width: Int) -> Int {
        ((height / patchSize) * (width / patchSize)) / (poolingKernelSize * poolingKernelSize)
    }

    /// Port of the Python Gemma4ImageProcessor's aspect-ratio preserving
    /// resize: the largest dimensions that (a) stay within the patch budget
    /// maxSoftTokens * poolingKernelSize², and (b) keep both sides divisible
    /// by poolingKernelSize * patchSize, so the pooling kernel is exact and
    /// the pooled grid covers the image fully at any aspect ratio.
    ///
    /// Note the config's `size` entry is deliberately ignored, as in the
    /// Python reference — models ship a vestigial 224x224 there.
    public func aspectPreservingTargetSize(for imageSize: CGSize) -> CGSize {
        let kernelArea = poolingKernelSize * poolingKernelSize
        let maxPatches = maxSoftTokens * kernelArea
        let sideMultiple = poolingKernelSize * patchSize
        let height = Double(imageSize.height)
        let width = Double(imageSize.width)

        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / max(height * width, 1)).squareRoot()
        var targetHeight = Int((factor * height / Double(sideMultiple)).rounded(.down))
        var targetWidth = Int((factor * width / Double(sideMultiple)).rounded(.down))

        // One side can floor to zero for extreme aspect ratios (both cannot:
        // their product is pinned near maxPatches, far above 1). Clamp it to
        // one pooling cell and cap the long side at the full token budget.
        let maxSideLength = maxSoftTokens
        if targetHeight == 0 {
            targetHeight = 1
            targetWidth = min(Int((width / max(height, 1)).rounded(.down)), maxSideLength)
            targetWidth = max(targetWidth, 1)
        } else if targetWidth == 0 {
            targetWidth = 1
            targetHeight = min(Int((height / max(width, 1)).rounded(.down)), maxSideLength)
            targetHeight = max(targetHeight, 1)
        }

        return CGSize(
            width: targetWidth * sideMultiple, height: targetHeight * sideMultiple)
    }
}
