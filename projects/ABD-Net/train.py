import torchreid
from ABD_components.abd_engine import ImageABDEngine
from torch import nn

torchreid.data.register_image_dataset('rock_dataset', torchreid.data.datasets.image.rock_dataset.RockDataSet)

datamanager = torchreid.data.ImageDataManager(
    root='/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/zhejiang_train',
    sources='rock_dataset',
    targets='rock_dataset',
    height=672,
    width=672,
    batch_size_train=16,
    batch_size_test=16,
    transforms=[]
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
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = ImageABDEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True,
    weight_t=0.1,
    weight_x=1,
    margin=1.2
)

engine.run(
    save_dir='log/abd_resnet50',
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False
)
