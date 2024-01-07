def Model_config():
    config = {}
    config['batch_size'] = 24       #32
    config['num_workers'] = 0                # 8 number of data loading workers
    config['epochs'] = 100                   # number of total epochs to run
    config['lr'] = 1e-4                       # initial learning rate
    config['num_classes'] = 86
    
    # Molormer
    config['num_layers'] = 3
    config['num_heads'] = 8

    config['hidden_dim'] = 256
    config['inter_dim'] = 256

    config['input_dropout_rate'] = 0
    config['encoder_dropout_rate'] = 0
    config['attention_dropout_rate'] = 0
    
    config['flatten_dim'] = 2048              # molormer:2048, transformer:8192， no_distil：8192， no_share:2048

    #amde
    config['message-passes']=2
    config['message-size']=25
    config['msg-depth']=2
    config['msg-hidden-dim']=50
    config['att-depth']=2
    config['att-hidden-dim']=50
    config['gather-width']=75
    config['gather-att-depth']=2
    config['gather-att-hidden-dim']=45
    config['gather-emb-depth']=2
    config['gather-emb-hidden-dim']=26

    config['out-depth']=2
    config['out-hidden-dim']=90
    config['out-layer-shrinkage']=0.6

    return config
