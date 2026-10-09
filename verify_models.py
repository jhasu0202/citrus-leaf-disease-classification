from app.core.model_loader import MODEL_CONFIGS, load_model
import torch
import traceback

print("=" * 60)
print("CITRUS DISEASE MODEL VERIFICATION")
print("=" * 60)

passed = []
failed = []

for name in MODEL_CONFIGS:
    print(f"\nTesting: {name}")

    try:
        model = load_model(name)
        model.eval()

        with torch.inference_mode():
            output = model(torch.randn(1, 3, 224, 224))

        assert output.shape == (1, 5), (
            f"Expected output shape (1, 5), got {tuple(output.shape)}"
        )

        print("PASS: Model loaded and returned five class scores.")
        passed.append(name)

        del model

    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        failed.append((name, str(exc)))

print("\n" + "=" * 60)
print(f"PASSED: {len(passed)}/{len(MODEL_CONFIGS)}")
print(f"FAILED: {len(failed)}/{len(MODEL_CONFIGS)}")

if failed:
    print("\nModels requiring fixes:")
    for name, error in failed:
        print(f"- {name}: {error}")

print("=" * 60)
