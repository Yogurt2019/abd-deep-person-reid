import torchreid
from projects.ABD_Net.ABD_components.abd_engine import ImageABDEngine
from torch import nn

torchreid.data.register_image_dataset('rock_dataset', torchreid.data.datasets.image.rock_dataset.RockDataSet)

datamanager = torchreid.data.ImageDataManager(
    root='/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/zhejiang_train',
    # root='/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/remote_data',
    sources='rock_dataset',
    targets='rock_dataset',
    height=672,
    width=672,
    batch_size_train=12,
    batch_size_test=12,
    transforms=['random_flip'],
    workers=32,
)

model = torchreid.models.build_model(
    name='abd_resnet',
    num_classes=datamanager.num_train_pids,
    loss='xent',
    pretrained=False
)

model = nn.DataParallel(model).cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.1
)


start_epoch = torchreid.utils.resume_from_checkpoint(
    'log/resnet50_abd/model.pth.tar-40',
    model,
    optimizer
)


scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=30
)

engine = ImageABDEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True,
    weight_t=0,
    weight_x=1,
    margin=1.2,
)

engine.run(
    save_dir='log/resnet50_abd/visualize',
    start_epoch=10,
    max_epoch=90,
    eval_freq=10,
    print_freq=10,
    test_only=True,
    fixbase_epoch=0,
    open_layers=['classifier'],
    visrank=True,
    visrank_topk=3
)
