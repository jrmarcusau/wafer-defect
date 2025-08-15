Files:
- cart_512.png : grayscale at 512
- cart_256.png : grayscale at 256
- edge512_canny/morph/(lsd).png : component edges at 512
- edge512_fused.png, overlay512_fused.png : fused edges at 512
- edge256_channel.png, overlay256_channel.png : DOWN-SAMPLED edge (what the model would see)
- coord_x_vis.png, coord_y_vis.png : coord channels visualization
- input_tensor_cartesian_4x256x256.npy/.pt : tensor [gray, edge, x, y]
- edge_channel_tensor_1x256x256.npy/.pt : edge-only tensor
