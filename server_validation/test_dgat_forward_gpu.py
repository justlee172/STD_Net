#!/usr/bin/env python3
"""GPU regression test for the full_v2 conDGAT forward path."""

from __future__ import annotations

import torch

from STD_Net.DGAT import conDGAT


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU regression test")

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = conDGAT(
        nfeat=4,
        nhid=8,
        common_hid_layers_num=2,
        fcnn_hid_layers_num=1,
        dropout=0.0,
        nout1=3,
        use_gatv2=True,
        gat_heads=4,
        output_transform="sparsemax",
        gated_fusion=True,
        reconstruction_gene_count=4,
    ).to(device)
    model.eval()

    features = torch.arange(20, dtype=torch.float32, device=device).reshape(5, 4)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
        ],
        dtype=torch.long,
        device=device,
    )
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
    graph = (edge_index, edge_weight)

    with torch.no_grad():
        output, layers = model(features, [graph, graph])

    if output.shape != (5, 3):
        raise AssertionError(f"unexpected output shape: {tuple(output.shape)}")
    torch.testing.assert_close(
        model.probabilities.sum(dim=1),
        torch.ones(5, device=device),
        atol=1e-6,
        rtol=1e-6,
    )
    if "gc_out111" not in layers:
        raise AssertionError("gc_out111 is missing from the returned layer mapping")
    if "gc_out112" in layers:
        raise AssertionError("invalid gc_out112 appeared in the returned layer mapping")

    print(
        "CUDA_FORWARD_OK",
        f"device={torch.cuda.get_device_name(device)}",
        f"shape={tuple(output.shape)}",
        f"layers={','.join(sorted(layers))}",
    )


if __name__ == "__main__":
    main()
