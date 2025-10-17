    def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
        for i, layer in enumerate(model):
            if i >= len(model) - num_to_forward_prefetch:
                break
            layers_to_prefetch = [
                model[i + j] for j in range(1, num_to_forward_prefetch + 1)
            ]
            layer.set_modules_to_forward_prefetch(layers_to_prefetch)
    
    def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
        for i, layer in enumerate(model):
            if i < num_to_backward_prefetch:
                continue
            layers_to_prefetch = [
                model[i - j] for j in range(1, num_to_backward_prefetch + 1)
            ]
            layer.set_modules_to_backward_prefetch(layers_to_prefetch)

    # todo:vit第一层不切，但挂hook prefetch后面层；切llm的embedding和llm_head
    set_modules_to_forward_prefetch(model.visual.blocks, num_to_forward_prefetch=2)
    model.visual.blocks[-1].set_modules_to_forward_prefetch([model.model.layers[0]])
    set_modules_to_forward_prefetch(model.model.layers, num_to_forward_prefetch=1)
    set_modules_to_backward_prefetch(model.visual.blocks, num_to_backward_prefetch=2)
    set_modules_to_backward_prefetch(model.model.layers, num_to_backward_prefetch=1)



return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]



    q_embed = torch_npu.npu_rotary_mul(q.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0)).squeeze(0)
    k_embed = torch_npu.npu_rotary_mul(k.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0)).squeeze(0)
