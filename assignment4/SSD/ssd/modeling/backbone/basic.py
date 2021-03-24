import torch
import torch.nn as nn

class DoubleConvReLU(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, kernel_size=3, padding=1, stride_2=2) -> None:
        super(DoubleConvReLU, self).__init__()
        self.layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, intermediate_channels, kernel_size, 1, padding),                             
            nn.ReLU(),                                      
            nn.Conv2d(intermediate_channels, out_channels, kernel_size, stride_2, padding)
        )
    
    def forward(self, x):
        return self.layer(x)


class BasicModel(nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.layers = nn.ModuleList()
        intermediate_channels = [128, 256, 128, 128, 128]

        self.layers.append(nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.output_channels[0], kernel_size=3, stride=2, padding=1)
        ))

        for i in range(4):
            self.layers.append(DoubleConvReLU(
                in_channels=self.output_channels[i],
                intermediate_channels=intermediate_channels[i],
                out_channels=self.output_channels[i+1],
            ))
        
        self.layers.append(nn.Conv2d(DoubleConvReLU(
                                    in_channels=self.output_channels[4],
                                    intermediate_channels=intermediate_channels[-1],
                                    out_channels=self.output_channels[5],
                                    padding=1,
                                    stride_2=1
        )))


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for layer in self.layers:
            x = layer(x)
            out_features.append(x)

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

