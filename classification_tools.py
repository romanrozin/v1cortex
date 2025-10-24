import numpy as np
import tensorflow as tf

import models

def create_model(network, input_population, bkg_weights, seq_len=100, n_input=10, n_output=2,
                 dtype=tf.float32, down_sampled_decode_noise_path=None,
                 input_weight_scale=1., gauss_std=.5, dampening_factor=.2, train_recurrent=True,
                 train_input=True, neuron_output=False, lRout_pop='all', L2_factor=.0,
                 return_state=False, down_sample=50, use_decoded_noise=False,
                 max_delay=5, batch_size=None, full_output=False, output_mode='garrett',
                 neuron_model='GLIF3', use_dale_law=True, scale=[1,1], _return_interal_variables=False,
                 ):
    x = tf.keras.layers.Input(shape=(seq_len, n_input,))
    neurons = network['n_nodes']

    if batch_size is None:
        batch_size = tf.keras.ops.shape(x)[0]
    else:
        batch_size = batch_size

    if neuron_model == 'GLIF3':
        cell = models.BillehColumn(network, input_population, bkg_weights,
                                   gauss_std=gauss_std, dampening_factor=dampening_factor,
                                   input_weight_scale=input_weight_scale,
                                   max_delay=max_delay, train_recurrent=train_recurrent, train_input=train_input, train_bkg=False,
                                   use_dale_law=use_dale_law, _return_interal_variables=_return_interal_variables)
    else:
        raise ValueError('Not supported neuron model!')

    zero_state = cell.zero_state(batch_size, dtype)
    
    # Keras 3 requires inputs to be flattened
    state_components = tf.nest.flatten(zero_state)
    component_shapes = [component.shape for component in state_components]
    input_layers = [
        tf.keras.layers.Input(
            batch_shape=(None,) + shape[1:],
            name=f'input_layer_{i}'
        )
        for i, shape in enumerate(component_shapes)
    ]

    # Pack back into nested structure for RNN
    initial_state_holder = tf.nest.pack_sequence_as(zero_state, input_layers)
    rnn_initial_state = initial_state_holder
    
    rnn_inputs = models.SparseLayer(
        cell.input_indices, cell.input_weight_values, cell.input_dense_shape,
        cell.bkg_weights, down_sampled_decode_noise_path=down_sampled_decode_noise_path,
        use_decoded_noise=use_decoded_noise, dtype=dtype, scale=scale, name='input_layer_sparse')(x)

    rnn_inputs = tf.keras.ops.cast(rnn_inputs, dtype)
   
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state, name='rsnn')
    out = rnn(rnn_inputs, initial_state=rnn_initial_state)[0]
    # constants=tf.zeros((batch_size,)), name='enable_state_gradients'

    # In Keras 3 RNN output is a concatenated tensor instead of a tuple
    spikes = out[..., :network['n_nodes']]
    voltage = out[..., network['n_nodes']:network['n_nodes']*2]
    states = out[..., network['n_nodes']*2:]
    print(spikes.shape)
    print(voltage.shape)
    print(states.shape)
    # rate = tf.keras.ops.cast(tf.keras.ops.mean(spikes, axis=(1, 2)), tf.float32)

    if neuron_output:
        indices = network['synapses']['indices']
        num_readout_synapses = 0
        num_synapses_per_neuron = np.zeros(32, np.int32)
        for post_ind, pre_ind in indices:
            if int(post_ind / 4) < 32:
                num_readout_synapses += 1
                num_synapses_per_neuron[int(post_ind / 4)] += 1
        print(f'> Readout synapses {num_readout_synapses}')
        print(f'> Synapses per readout neuron {np.mean(num_synapses_per_neuron):.1f}')
        
        output_spikes = 1 / dampening_factor * spikes + (1 - 1 / dampening_factor) * tf.keras.ops.stop_gradient(spikes)
        scale = (1 + tf.nn.softplus(tf.keras.layers.Dense(1)(tf.zeros((1, 1)))))
        if output_mode == 'garrett':
            output = tf.keras.layers.Lambda(lambda x: tf.gather(x, network['localized_readout_neuron_ids_0'], axis=2))(output_spikes)
            output = tf.keras.ops.mean(output, -1)
            # thresh = tf.keras.layers.Dense(1)(tf.keras.ops.zeros_like(output))
            thresh = tf.keras.ops.zeros_like(output) + .01
            output = tf.keras.ops.stack([thresh[..., 0], output[..., -1]], -1) * scale
            # output *= (1 + tf.nn.softplus(tf.keras.layers.Dense(1)(tf.keras.ops.zeros_like(spikes[..., :1]))))
        elif output_mode == 'vcd_grating':
            output = tf.keras.layers.Lambda(lambda x: tf.gather(x, network['localized_readout_neuron_ids_1'], axis=2))(output_spikes)
            output = tf.keras.ops.mean(output, -1)
            thresh = tf.keras.ops.zeros_like(output) + .01
            output = tf.keras.ops.stack([thresh[..., 0], output[..., -1]], -1) * scale
        elif output_mode == 'ori_diff':
            output = tf.keras.layers.Lambda(lambda x: tf.gather(x, network['localized_readout_neuron_ids_2'], axis=2))(output_spikes)
            output = tf.keras.ops.mean(output, -1)
            thresh = tf.keras.ops.zeros_like(output) + .01
            output = tf.keras.ops.stack([thresh[..., 0], output[..., -1]], -1) * scale

        elif output_mode == 'evidence':
            output1 = tf.keras.layers.Lambda(lambda x: tf.gather(x, network['localized_readout_neuron_ids_3'], axis=2))(output_spikes)
            output2 = tf.keras.layers.Lambda(lambda x: tf.gather(x, network['localized_readout_neuron_ids_4'], axis=2))(output_spikes)
            output1 = tf.keras.ops.mean(output1, -1)
            output2 = tf.keras.ops.mean(output2, -1)
            output = tf.keras.ops.stack([output1[..., -1], output2[..., -1]], -1) * scale
        elif output_mode == '10class':
            outputs = []
            for i in range(10):
                t_output = tf.keras.layers.Lambda(lambda x: tf.gather(x, network[f'localized_readout_neuron_ids_{i + 5}'], axis=2))(output_spikes)
                t_output = tf.keras.ops.mean(t_output, -1)
                outputs.append(t_output)
            output = tf.keras.layers.Lambda(lambda x: tf.concat(x, -1) * scale)(outputs)
        else:
            raise ValueError(f'Unrecognized output_mode: {output_mode}')
    else:
        if lRout_pop != 'all':
            out_pop_spikes = tf.keras.layers.Lambda(lambda x: tf.gather(x, network['laminar_indices'][lRout_pop], axis=2))(spikes)
        else:
            out_pop_spikes = spikes

        # use recurrent weight's mean, otherwise the learning rate is inconsistent for both
        linear_readout_initializer = 'glorot_uniform'#tf.keras.initializers.RandomNormal(mean=0, stddev=0.008, seed=None) # 0.008 = 2/sqrt(51978)

        output_all = tf.keras.layers.Dense(18, name='projection', trainable=True,
                    kernel_regularizer=tf.keras.regularizers.l2(L2_factor), kernel_initializer=linear_readout_initializer)(out_pop_spikes)
        if output_mode == 'garrett':
            output = tf.keras.layers.Lambda(lambda x: tf.gather(x,[0,1],axis=2))(output_all)
        elif output_mode == 'vcd_grating':
            output = tf.keras.layers.Lambda(lambda x: tf.gather(x,[2,3],axis=2))(output_all)
        elif output_mode == 'ori_diff':
            output = tf.keras.layers.Lambda(lambda x: tf.gather(x,[4,5],axis=2))(output_all)
        elif output_mode == 'evidence':
            output = tf.keras.layers.Lambda(lambda x: tf.gather(x,[6,7],axis=2))(output_all)
        elif output_mode == '10class':
            output = tf.keras.layers.Lambda(lambda x: tf.gather(x,tf.range(8,18,1),axis=2))
        else:
            raise ValueError(f'Unrecognized output_mode: {output_mode}')

    output = tf.keras.layers.Lambda(lambda _a: _a, name='prediction')(output)

    mean_output = tf.keras.ops.reshape(output, (-1, int(seq_len / down_sample), down_sample, n_output))
    mean_output = tf.keras.ops.mean(mean_output, 2)
    mean_output = tf.keras.ops.softmax(mean_output, axis=-1)    

    if full_output:
        outputs = [mean_output, spikes, voltage, states]
    else:
        outputs = mean_output

    # Use flattened inputs as required by Keras 3
    many_input_model = tf.keras.Model(inputs=[x] + input_layers, outputs=outputs)
    
    
    return many_input_model, component_shapes
