import torch

class Predictor:

    def __init__(self, model):
        
        self.model = model
        self.images=None
        self.features = None
        self.mask_features = None
        self.multi_scale_features=None
        self.pred_masks = None

    def get_prediction(self, clicker):
        if self.features is None:
            (processed_results, outputs, self.images,
            _, self.features, self.mask_features,
            self.multi_scale_features, _, _,_) = self.model(clicker.inputs, max_timestamp=clicker.max_timestamps)

        else:
            out = self.model(clicker.inputs, self.images, clicker.num_insts,
                        self.features, self.mask_features,
                        self.multi_scale_features,
                        clicker.num_clicks_per_object,
                        clicker.fg_coords, clicker.bg_coords,
                        max_timestamp = clicker.max_timestamps
                  )
            processed_results = out[0]
        pred_masks = processed_results[0]['instances'].pred_masks.to('cpu',dtype=torch.uint8)
       
        return pred_masks
        
