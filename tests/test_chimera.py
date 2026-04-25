import pytest
torch = pytest.importorskip("torch")

from chimera import Chimera51ForCausalLM, ChimeraTokenizer, load_config, pack_ternary, scale_config, unpack_ternary
from chimera.inference import SpanBank
from chimera.moe import MoELayer
from chimera.quantization import BitLinear, ternarize_weight


def cfg():
    c = scale_config(load_config("config.json"), "nano")
    c["vocab_size"] = 512
    c["span_inference"]["enabled"] = False
    return c


def test_pack_unpack_roundtrip():
    q = torch.tensor([[-1, 0, 1, 1, -1]], dtype=torch.int8)
    packed = pack_ternary(q)
    out = unpack_ternary(packed, q.shape[-1], dtype=torch.float32).to(torch.int8)
    assert torch.equal(q, out)


def test_bitlinear_forward_backward_and_packed():
    layer = BitLinear(7, 5)
    x = torch.randn(3, 7, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert x.grad is not None
    layer.prepare_for_inference(); layer.eval()
    with torch.no_grad():
        out = layer(torch.randn(2, 7))
    assert out.shape == (2, 5)


def test_model_forward_loss_and_generate_shape():
    model = Chimera51ForCausalLM(cfg())
    x = torch.randint(0, 512, (2, 8))
    y = torch.randint(0, 512, (2, 8))
    out = model(x, labels=y)
    assert out.logits.shape == (2, 8, 512)
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_moe_and_span_bank_shapes():
    moe = MoELayer(32, 64, n_routed_experts=3, n_shared_experts=1, num_experts_per_tok=2)
    x = torch.randn(2, 4, 32)
    assert moe(x).shape == x.shape
    bank = SpanBank(max_entries=8, hidden_size=32)
    bank.add(torch.randn(3, 32), torch.randn(3, 32))
    assert bank.query(torch.randn(5, 32)).shape == (5, 32)


def test_tokenizer_fallback_roundtrip():
    tok = ChimeraTokenizer(vocab_size=512)
    text = "hello cpu"
    assert tok.decode(tok.encode(text)) == text
