import numpy as np
import importlib.machinery
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
sys.path.append('..')
from network import Snow_loss, Snow_dataset, Snow_model

from ignite.metrics import Accuracy, Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, Engine
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.handlers import Checkpoint, DiskSaver

def main():
    torch.manual_seed(42)
    def data_prepare(batch):
        z, y, x = batch
        return z.cuda(), y.cuda(), x.cuda()
    def customized_trainer(model, optimizer, loss_fn, prepare_batch,
                                  device='cuda', non_blocking=False):
            if device:
                model.to(device)
                loss_fn.to(device)
            def process_function(engine, batch):
                optimizer.zero_grad()
                model.train()
                z, y, x = prepare_batch(batch)
                y_hat, y_prime, z_hat = model(x)
                ''' 
                Do weight regularization as suggested in the paper
                As we know, Adam optimizer's weight decay is different from l2 regularization.
                '''
                weight_reg = 0
                if weight_r:
                    for p in model.parameters():
                        weight_reg = weight_reg + p.square().sum()
                loss = loss_fn(y_hat, y_prime, z_hat, y, z, weight_reg)
                loss.backward()
                optimizer.step()
                return loss.item()
    
            return Engine(process_function)
    def customized_evaluator(model, loss_fn, prepare_batch, metrics=None,
                                    device='cuda', non_blocking=False):
            metrics = metrics or {}

            if device:
                model.to(device)
                loss_fn.to(device)

            def _inference(engine, batch):
                model.eval()
                with torch.no_grad():
                    optimizer.zero_grad()
                    model.train()
                    z, y, x = prepare_batch(batch)
                    y_hat, y_prime, z_hat = model(x, train=False)
                    ''' 
                    Do weight regularization as suggested in the paper
                    As we know, Adam optimizer's weight decay is different from l2 regularization.
                    '''
                    weight_reg = 0
                    if weight_r:
                        for p in model.parameters():
                            weight_reg = weight_reg + p.square().sum()
                    return y_hat, y_prime, {"z_hat":z_hat, "y":y, "z":z, "weight_reg":weight_reg}
            engine = Engine(_inference)

            for name, metric in metrics.items():
                metric.attach(engine, name)

            return engine

    def training_log(output):
        loss = output
        result = dict()
        result["Loss"] = loss
        return result




    run_dir = './'
    run_name = 'first_try_desnow_weightdecay'
    tensorboard_name = './test_tb'
    num_epochs = 1000
    snapshot_freq = 10
    dryrun = False
    batch_size = 5
    device='cuda'
    load_model=False
    model_path = './first_try_desnow_weightdecay/best_checkpoint_-0.1485.pt'
    weight_r = False
    weight_decay = True
    
    
    '''
    xavier initialization as suggested in the paper
    '''
    def init_normal(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            random.seed(datetime.now())
            seed_number = random.randint(0, 100)
            random.seed(0)
            torch.manual_seed(seed_number)
            if type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
            torch.manual_seed(torch.initial_seed())

    model = Snow_model(initial_size=16).to(device)
    model.apply(init_normal)
    
    if load_model:
        cp = torch.load(model_path)
        model.load_state_dict(cp['model_state_dict'])
    
    trainset = Snow_dataset('./all/')
    validateset = Snow_dataset('./all/', train=False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(validateset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    if weight_decay:
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=5e-4)
    criterion = Snow_loss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)

    print(f"training set length {len(trainset)} ")
    print(f"validation set length {len(validateset)} ")

    trainer = customized_trainer(model, optimizer, criterion, data_prepare)

    val_metrics = {
        "loss" : Loss(criterion)
    }      
    evaluator = customized_evaluator(model, criterion, data_prepare, metrics=val_metrics)

    # setup dir
    train_dir = os.path.join(run_dir, run_name)
    log_dir = os.path.join(tensorboard_name, run_name)
    n = 1
    while os.path.exists(log_dir):
        run_name = f'{run_name}{n}'
        log_dir = os.path.join(tensorboard_name, run_name)
        n += 1
    print(f'Unique Name: {run_name}')


    @trainer.on(Events.ITERATION_COMPLETED(every=50))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {trainer.state.epoch} Avg loss: {metrics['loss']}")

    @evaluator.on(Events.ITERATION_COMPLETED(every=100))
    def log_eval_process(engine):
        print(engine.state.iteration)

    # handler for interrupt exception
    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            print("KeyboardInterrupt caught. Exiting gracefully.")
        else:
            raise e
            print("Try to restart")


    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: scheduler.step(evaluator.state.metrics['loss']))
        else:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: scheduler.step(evaluator.state.metrics['loss']))

    def score_function(engine):
        return 0 - engine.state.metrics['loss']

    to_save = {
        'model_state_dict': model,
        'optimizer_state_dict': optimizer,
        'scheduler_state_dict': scheduler
    }
    checkpoint_handler = Checkpoint(to_save, DiskSaver(train_dir, create_dir=True, require_empty=False),filename_prefix="snapshot", n_saved=None)
    model_checkpoint_handler = Checkpoint(to_save, DiskSaver(train_dir, create_dir=True, require_empty=False), "best", n_saved=1, score_function=score_function)
    
    evaluator.add_event_handler(
        event_name=Events.EPOCH_COMPLETED(every=1), handler=model_checkpoint_handler
    )
    
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED(every=snapshot_freq), handler=checkpoint_handler
    )

    if not dryrun:
        tb_logger = TensorboardLogger(log_dir=log_dir)
        tb_logger.attach_output_handler(
            trainer,
            tag="Overall For Training", 
            output_transform=training_log,
            global_step_transform=global_step_from_engine(trainer),
            event_name=Events.ITERATION_COMPLETED(every=50),
        )

        tb_logger.attach_output_handler(
            evaluator,
            tag="Overall For Validation", 
            metric_names = "all",
            global_step_transform=global_step_from_engine(trainer),
            event_name=Events.EPOCH_COMPLETED,
        )

        tb_logger.attach_opt_params_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED(every=100),
            optimizer=optimizer,
            tag="Parameters",
            param_name='lr'  # optional
        )

    trainer.run(train_loader, max_epochs=num_epochs)
    tb_logger.close()


if __name__ == '__main__':
    main()