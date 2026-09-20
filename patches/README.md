# patches

## cosmos-transfer2.5-rtx3090.patch

Applies to a [cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
checkout (tested at `2ff49d0`) and is what `carla-transfer prepare-scene`
(`--first-image transfer25`) needs on a 24 GB GPU:

- `predict2/text_encoders/text_encoder.py`,
  `transfer2/inference/inference_pipeline.py`,
  `transfer2_multiview/inference/inference.py`,
  `predict2_multiview/models/multiview_vid2vid_model_rectified_flow.py`:
  with `COSMOS_EVICT_TEXT_ENCODER=1` the 7B Cosmos-Reason1 text encoder and
  the 2B denoiser never share the card — the denoiser is parked on the host
  while prompts are embedded (including the empty/negative prompts computed
  in the model's `__init__`), and the encoder goes back to the host after.
- `imaginaire/auxiliary/world_scenario/utils/pcd_utils.py`: triangulate
  polygons on the two axes with the largest extent. Synthetic maps (roadgen)
  have exactly flat ground; after the loader's FLU→RDF conversion every
  intersection area had constant y and Delaunay on x-y raised
  `QhullError: Initial simplex is flat`.

```bash
cd ~/cosmos-transfer2.5 && git apply /path/to/carla_transfer/patches/cosmos-transfer2.5-rtx3090.patch
```
