"""Architecture tests for the Transformer in nstudy.core.model.

The reference is Vaswani et al., "Attention Is All You Need" (NeurIPS 2017).
Section and table numbers below refer to that paper.
"""

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from nstudy.core.model import (
    DecoderLayer,
    EncoderLayer,
    MultiHeadAttention,
    Transformer,
    sinusoidal_positions,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# Section 5.1: a shared source-target vocabulary of about 37000 tokens.
VOCAB_SIZE = 37_000

# Table 3.
BASE = dict(d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1)
BIG = dict(d_model=1024, num_heads=16, num_layers=6, d_ff=4096, dropout=0.3)

# Exact counts for a 37000-token vocabulary with the embedding matrix shared
# three ways. The paper's Table 3 values are in comments.
BASE_PARAMS = 63_082_496  # 65M
BIG_PARAMS = 214_245_376  # 213M


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def modules_of(model: nn.Module, kind: type) -> list:
    return [m for m in model.modules() if isinstance(m, kind)]


def small_model(**overrides) -> Transformer:
    kwargs = dict(vocab_size=100, d_model=64, num_heads=4, num_layers=2, d_ff=128)
    kwargs.update(overrides)
    return Transformer(**kwargs).eval()


def random_tokens(batch: int, length: int, vocab_size: int = 100) -> torch.Tensor:
    """Random token ids that never include the padding id 0."""
    return torch.randint(1, vocab_size, (batch, length))


# ---------------------------------------------------------------------------
# Size (Table 3)
# ---------------------------------------------------------------------------


def test_defaults_are_the_base_model():
    model = Transformer(VOCAB_SIZE)
    assert count_params(model) == BASE_PARAMS
    assert len(model.encoder) == len(model.decoder) == 6
    assert model.embedding.embedding_dim == 512
    assert {m.p for m in modules_of(model, nn.Dropout)} == {0.1}


@pytest.mark.parametrize(
    "hparams, exact, paper", [(BASE, BASE_PARAMS, 65e6), (BIG, BIG_PARAMS, 213e6)]
)
def test_param_count(hparams, exact, paper):
    count = count_params(Transformer(VOCAB_SIZE, **hparams))
    assert count == exact
    assert count == pytest.approx(paper, rel=0.05)


@pytest.mark.parametrize("hparams", [BASE, BIG])
def test_layer_dimensions(hparams):
    model = Transformer(1000, **hparams)
    d_model, d_ff = hparams["d_model"], hparams["d_ff"]
    for layer in [*model.encoder, *model.decoder]:
        assert (layer.linear1.in_features, layer.linear1.out_features) == (d_model, d_ff)
        assert (layer.linear2.in_features, layer.linear2.out_features) == (d_ff, d_model)
    for attention in modules_of(model, MultiHeadAttention):
        assert attention.num_heads == hparams["num_heads"]
        assert attention.head_dim == 64  # d_k = d_v = d_model / h = 64 in both models


def test_d_model_must_be_divisible_by_num_heads():
    with pytest.raises(ValueError):
        Transformer(100, d_model=100, num_heads=8)


# ---------------------------------------------------------------------------
# Layer structure (Section 3.1)
# ---------------------------------------------------------------------------


def test_encoder_layer_has_two_sublayers_and_decoder_layer_has_three():
    model = small_model()
    for layer in model.encoder:
        assert isinstance(layer, EncoderLayer)
        assert len(modules_of(layer, MultiHeadAttention)) == 1
        assert len(modules_of(layer, nn.LayerNorm)) == 2
    for layer in model.decoder:
        assert isinstance(layer, DecoderLayer)
        assert len(modules_of(layer, MultiHeadAttention)) == 2
        assert len(modules_of(layer, nn.LayerNorm)) == 3


def test_there_is_one_layer_norm_per_sublayer_and_none_elsewhere():
    """Post-norm needs no extra norm on the embeddings or the stack outputs."""
    model = Transformer(1000, **BASE)
    assert len(modules_of(model, nn.LayerNorm)) == 6 * 2 + 6 * 3


def test_sublayer_output_is_layer_norm_of_residual_sum():
    """LayerNorm(x + Sublayer(x)): a layer's output is normalized at every position."""
    model = small_model()
    x = torch.randn(2, 7, 64) * 5 + 3
    memory = torch.randn(2, 5, 64)
    outputs = [model.encoder[0](x, None), model.decoder[0](x, memory, None, None)]
    for out in outputs:
        assert torch.allclose(out.mean(-1), torch.zeros(2, 7), atol=1e-5)
        assert torch.allclose(out.std(-1, unbiased=False), torch.ones(2, 7), atol=1e-3)


def test_norm_layer_is_pluggable():
    model = small_model(norm_layer=lambda d_model: nn.RMSNorm(d_model))
    assert not modules_of(model, nn.LayerNorm)
    assert len(modules_of(model, nn.RMSNorm)) == 2 * 2 + 2 * 3


# ---------------------------------------------------------------------------
# Pre-norm option (norm_first=True); not in the paper
# ---------------------------------------------------------------------------


def silence_sublayers(layer: nn.Module) -> None:
    """Zero the last linear map of every sub-layer so each sub-layer outputs zero."""
    with torch.no_grad():
        for module in [*modules_of(layer, MultiHeadAttention)]:
            module.out_proj.weight.zero_()
            module.out_proj.bias.zero_()
        layer.linear2.weight.zero_()
        layer.linear2.bias.zero_()


@pytest.mark.parametrize("norm_first", [False, True])
def test_residual_path_skips_the_norm_only_with_pre_norm(norm_first):
    """x + Sublayer(norm(x)) leaves x untouched when the sub-layers output zero."""
    model = small_model(norm_first=norm_first)
    x = torch.randn(2, 7, 64) * 5 + 3
    memory = torch.randn(2, 5, 64)
    encoder_layer, decoder_layer = model.encoder[0], model.decoder[0]
    silence_sublayers(encoder_layer)
    silence_sublayers(decoder_layer)
    outputs = [encoder_layer(x, None), decoder_layer(x, memory, None, None)]
    for out in outputs:
        assert torch.equal(out, x) == norm_first


def test_post_norm_has_no_final_norms():
    model = small_model()
    assert model.encoder_norm is None
    assert model.decoder_norm is None


def test_pre_norm_adds_one_final_norm_per_stack():
    model = Transformer(VOCAB_SIZE, **BASE, norm_first=True)
    assert isinstance(model.encoder_norm, nn.LayerNorm)
    assert isinstance(model.decoder_norm, nn.LayerNorm)
    assert len(modules_of(model, nn.LayerNorm)) == 6 * 2 + 6 * 3 + 2
    assert count_params(model) == BASE_PARAMS + 2 * (2 * 512)


def test_pre_norm_encoder_output_is_normalized():
    model = small_model(norm_first=True)
    memory = model.encode(random_tokens(2, 9))
    assert torch.allclose(memory.mean(-1), torch.zeros(2, 9), atol=1e-5)
    assert torch.allclose(memory.std(-1, unbiased=False), torch.ones(2, 9), atol=1e-3)


def test_pre_norm_final_norms_use_norm_layer():
    model = small_model(norm_first=True, norm_layer=lambda d_model: nn.RMSNorm(d_model))
    assert isinstance(model.encoder_norm, nn.RMSNorm)
    assert isinstance(model.decoder_norm, nn.RMSNorm)
    assert not modules_of(model, nn.LayerNorm)


def torch_reference_stacks(norm_first: bool) -> tuple[nn.Module, nn.Module]:
    """PyTorch's own encoder and decoder stacks, sized like ``small_model``."""
    kwargs = dict(dropout=0.0, batch_first=True, norm_first=norm_first)
    final_norm = lambda: nn.LayerNorm(64) if norm_first else None  # noqa: E731
    encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(64, 4, 128, **kwargs),
        num_layers=2,
        norm=final_norm(),
        enable_nested_tensor=False,
    )
    decoder = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(64, 4, 128, **kwargs), num_layers=2, norm=final_norm()
    )
    # The stacks are built from copies of one layer; make every weight distinct.
    for parameter in [*encoder.parameters(), *decoder.parameters()]:
        nn.init.normal_(parameter, std=0.2)
    return encoder.eval(), decoder.eval()


def torch_to_our_state_dict(encoder: nn.Module, decoder: nn.Module) -> dict:
    state = {}
    for prefix, stack in (("encoder", encoder), ("decoder", decoder)):
        for key, value in stack.state_dict().items():
            if key.startswith("norm."):
                key = key.replace("norm.", f"{prefix}_norm.")
            else:
                key = key.replace("layers.", f"{prefix}.").replace("multihead_attn", "cross_attn")
            if "in_proj_" in key:
                # PyTorch packs the query, key and value projections into one tensor.
                kind = key.rsplit("_", 1)[1]
                for name, chunk in zip("qkv", value.chunk(3)):
                    state[key.replace(f"in_proj_{kind}", f"{name}_proj.{kind}")] = chunk
            else:
                state[key] = value
    return state


@pytest.mark.parametrize("norm_first", [False, True])
def test_outputs_match_pytorch_transformer_given_the_same_weights(norm_first):
    encoder, decoder = torch_reference_stacks(norm_first)
    model = small_model(norm_first=norm_first)
    state = torch_to_our_state_dict(encoder, decoder)
    state["embedding.weight"] = state["generator.weight"] = model.embedding.weight
    model.load_state_dict(state, strict=True)

    src, tgt = random_tokens(3, 12), random_tokens(3, 9)
    src[1, 8:] = 0  # padding at the end of one source sentence
    future = torch.ones(9, 9, dtype=torch.bool).triu(diagonal=1)
    with torch.no_grad():
        memory = encoder(model.embed(src), src_key_padding_mask=src == 0)
        hidden = decoder(
            model.embed(tgt), memory, tgt_mask=future, memory_key_padding_mask=src == 0
        )
        expected = model.generator(hidden)
        assert torch.allclose(model(src, tgt), expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Embeddings and positions (Sections 3.4 and 3.5)
# ---------------------------------------------------------------------------


def test_embeddings_and_pre_softmax_projection_share_one_weight_matrix():
    model = small_model()
    assert model.generator.weight is model.embedding.weight
    assert model.generator.bias is None


def test_positional_encoding_follows_the_sinusoid_formula():
    d_model = 16
    table = sinusoidal_positions(50, d_model)
    assert table.shape == (50, d_model)
    for pos in (0, 1, 7, 49):
        for i in (0, 3, 7):
            angle = pos / 10000 ** (2 * i / d_model)
            assert table[pos, 2 * i].item() == pytest.approx(math.sin(angle), abs=1e-5)
            assert table[pos, 2 * i + 1].item() == pytest.approx(math.cos(angle), abs=1e-5)


def test_embeddings_are_scaled_by_sqrt_d_model_before_adding_positions():
    model = small_model()
    tokens = random_tokens(2, 9)
    expected = model.embedding(tokens) * math.sqrt(64) + sinusoidal_positions(9, 64)
    assert torch.allclose(model.embed(tokens), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_output_is_one_distribution_over_the_vocabulary_per_target_position():
    model = small_model()
    logits = model(random_tokens(3, 11), random_tokens(3, 8))
    assert logits.shape == (3, 8, 100)


def test_decoder_cannot_see_subsequent_positions():
    model = small_model()
    src, tgt = random_tokens(2, 6), random_tokens(2, 8)
    changed = tgt.clone()
    changed[:, 5:] = random_tokens(2, 3)
    with torch.no_grad():
        before, after = model(src, tgt), model(src, changed)
    assert torch.allclose(before[:, :5], after[:, :5], atol=1e-6)
    assert not torch.allclose(before[:, 5:], after[:, 5:], atol=1e-6)


def test_source_padding_does_not_change_the_output():
    model = small_model()
    src, tgt = random_tokens(2, 6), random_tokens(2, 8)
    padded = torch.cat([src, torch.zeros(2, 4, dtype=torch.long)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(src, tgt), model(padded, tgt), atol=1e-5)


# ---------------------------------------------------------------------------
# Hydra configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_params",
    [("transformer_base", BASE_PARAMS), ("transformer_big", BIG_PARAMS)],
)
def test_hydra_config_builds_the_requested_model(name, expected_params):
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="config", overrides=[f"model={name}"])
    assert count_params(hydra.utils.instantiate(cfg.model)) == expected_params


def test_hydra_config_can_switch_to_pre_norm():
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="config", overrides=["model=transformer_base", "model.norm_first=true"]
        )
    model = hydra.utils.instantiate(cfg.model)
    assert all(layer.norm_first for layer in [*model.encoder, *model.decoder])
    assert model.encoder_norm is not None


# ---------------------------------------------------------------------------
# Comparison with Hugging Face's MarianMTModel
# ---------------------------------------------------------------------------

# Hugging Face name -> ours. Order matters: longer names first.
HF_RENAMES = [
    ("model.shared.", "embedding."),
    ("model.encoder.layers.", "encoder."),
    ("model.decoder.layers.", "decoder."),
    ("encoder_attn_layer_norm.", "norm2."),
    ("self_attn_layer_norm.", "norm1."),
    ("encoder_attn.", "cross_attn."),
    ("fc1.", "linear1."),
    ("fc2.", "linear2."),
]


def marian(vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int):
    transformers = pytest.importorskip("transformers")
    config = transformers.MarianConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=num_layers,
        decoder_layers=num_layers,
        encoder_attention_heads=num_heads,
        decoder_attention_heads=num_heads,
        encoder_ffn_dim=d_ff,
        decoder_ffn_dim=d_ff,
        activation_function="relu",
        scale_embedding=True,
        share_encoder_decoder_embeddings=True,
        pad_token_id=0,
        decoder_start_token_id=0,
        max_position_embeddings=64,
    )
    return transformers.MarianMTModel(config).eval()


def to_our_state_dict(hf_model) -> dict:
    """Rename a Marian state dict to ours, dropping tied copies and position tables."""
    skipped = ("embed_tokens", "embed_positions", "lm_head", "final_logits_bias")
    state = {}
    for key, value in hf_model.state_dict().items():
        if any(name in key for name in skipped):
            continue
        for theirs, ours in HF_RENAMES:
            key = key.replace(theirs, ours)
        if key.startswith("encoder."):
            key = key.replace("final_layer_norm.", "norm2.")
        elif key.startswith("decoder."):
            key = key.replace("final_layer_norm.", "norm3.")
        state[key] = value
    state["generator.weight"] = state["embedding.weight"]
    return state


def test_param_count_matches_hugging_face_marian():
    hf_model = marian(VOCAB_SIZE, d_model=512, num_heads=8, num_layers=6, d_ff=2048)
    assert count_params(hf_model) == BASE_PARAMS


def test_outputs_match_hugging_face_marian_given_the_same_weights():
    hf_model = marian(100, d_model=64, num_heads=4, num_layers=2, d_ff=128)
    model = small_model()
    model.load_state_dict(to_our_state_dict(hf_model), strict=True)

    # Marian lays its sinusoid table out as [all sines | all cosines]; the paper
    # interleaves them. Give Marian the paper's table so only the layers are compared.
    with torch.no_grad():
        for stack in (hf_model.model.encoder, hf_model.model.decoder):
            stack.embed_positions.weight.copy_(sinusoidal_positions(64, 64))

    src, tgt = random_tokens(3, 12), random_tokens(3, 9)
    src[1, 8:] = 0  # padding at the end of one source sentence
    with torch.no_grad():
        ours = model(src, tgt)
        theirs = hf_model(
            input_ids=src, attention_mask=(src != 0).long(), decoder_input_ids=tgt
        ).logits
    assert torch.allclose(ours, theirs, atol=1e-4)
