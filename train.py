from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt

from dataset import *
from function import *
from model import * 
from transforms import *
from preprocess import *

def avg_train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, lr_num, hidden_size, num_layers, fold_num , avg_loss, vstr_loss, avg_tr_score, avg_vl_score, avg_diff_score, args, folder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_score = 0.
    setup_logger('./log_dir')
    gt_column = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
    weights = {
        'Q1': 1.5, 'Q2': 1.5, 'Q3': 1, 'S1': 1.5, 'S2': 1.5, 'S3': 1.5, 'S4': 1.5
    }
    
    seed_everything(1004)
    logging.info(f'Fold {fold_num} | Learning rate : {lr_num} | Hidden size : {hidden_size} | Num layers : {num_layers} | epochs : {epochs}')

    best_val_loss = float('inf')

    tr_loss = []
    vl_loss = []
    tr_score = []
    vl_score = []
    diff_score = []
    
    for epoch in range(epochs):
        model.train()
        loss_arr = []
        score = 0
        
        for _, (inputs, labels, length) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, length) 
            loss = criterion(outputs, labels)
            loss.backward()
            loss_arr.append(loss.item())
             
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            
            optimizer.step()
            
            output = (outputs >= args.threshold).int()
            output = output.cpu()
            label = labels.cpu()
            f1_scores = [f1_score(label[:, i].numpy(), output[:, i].numpy(), average='macro') for i in range(len(gt_column))]
            weighted_scores = sum([f1 * weights[label] for label, f1 in zip(gt_column, f1_scores)])
            score += weighted_scores

        result = score / len(train_loader)
        average_loss = sum(loss_arr) / len(loss_arr)
        tr_loss.append(average_loss)
        tr_score.append(result)
        logging.info(f'Fold {fold_num} | Train | Epoch {epoch+1}/{epochs}, Loss: {average_loss}, Score: {result}')
        print(f'Fold {fold_num} | Train | Epoch {epoch+1}/{epochs}, Loss: {average_loss}, Score: {result}')

        # Validation
        
        model.eval()
        val_loss_arr = []
        val_score = 0

        with torch.no_grad():
            for val_inputs, val_labels, val_length in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                
                val_outputs = model(val_inputs, val_length)
                
                val_loss = criterion(val_outputs, val_labels)
                
                val_output = (val_outputs >= args.threshold).int()
                
                val_output = val_output.cpu()
                val_label = val_labels.cpu()
                
                val_f1_scores = [f1_score(val_label[:, i].numpy(), val_output[:, i].numpy(), average='macro') for i in range(len(gt_column))]
                val_weighted_scores = sum([f1 * weights[label] for label, f1 in zip(gt_column, val_f1_scores)])
                
                val_score += val_weighted_scores
                val_loss_arr.append(val_loss.item())
                
        val_result = val_score / len(val_loader)
        val_average_loss = sum(val_loss_arr) / len(val_loss_arr)
        
        vl_loss.append(val_average_loss)
        vl_score.append(val_result)
        diff_score.append(result - val_result)
        
        logging.info(f'Fold {fold_num} | Validation | Epoch {epoch+1}/{epochs}, Loss: {val_average_loss}, Score: {val_result}')
        print(f'Fold {fold_num} | Validation | Epoch {epoch+1}/{epochs}, Loss: {val_average_loss}, Score: {val_result}')
        
        scheduler.step()

        if val_average_loss < best_val_loss:
            best_val_loss = val_average_loss
            best_model_wts = model.state_dict().copy()
            best_score = val_result

    model.load_state_dict(best_model_wts)
    epoch_lst = list(range(0, epochs))
    logging.info(f'Fold {fold_num} | Best score {best_score}')

    vstr_loss.append(tr_loss)
    avg_loss.append(vl_loss)
    avg_tr_score.append(tr_score)
    avg_vl_score.append(vl_score)
    avg_diff_score.append(diff_score)

    print(f"Best score {best_score}")
    logging.info(f"Best score {best_score}")
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_lst, tr_loss, label='Training Loss')
    plt.plot(epoch_lst, vl_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(tr_score, label='Training Score')
    plt.plot(vl_score, label='Validation Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training and Validation Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{folder_path}/fold{fold_num}.png')  # 그래프를 파일로 저장
    plt.close()

def full_train(model, train_dataset, train_loader, criterion, optimizer, scheduler, epochs, lr_num, hidden_size, num_layers, avg_loss, vstr_loss, avg_tr_score, avg_vl_score, args):
    
    best_score = 0.
    
    setup_logger('./log_dir')
    
    gt_column = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
    weights = {
        'Q1': 1.5, 'Q2': 1.5, 'Q3': 1, 'S1': 1.5, 'S2': 1.5, 'S3': 1.5, 'S4': 1.5
    }
    
    seed_everything(1004)
    logging.info(f'Train | Learning rate : {lr_num} | hidden size : {hidden_size} | num layers : {num_layers} | epochs : {epochs}')

    tr_loss = []
    vl_loss = []
    
    tr_score = []
    vl_score = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Full Train
    
    for epoch in range(epochs):
        model.train()  
        loss_arr = []
        score = 0

        for i, (inputs, labels, length) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, length) 
            loss = criterion(outputs, labels)
        
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)

            optimizer.step()
            
            loss_arr.append(loss.item())
            output = (outputs >= args.threshold).int()
            output = output.cpu()
            label = labels.cpu()
            f1_scores = [f1_score(label[:, i].numpy(), output[:, i].numpy(), average='macro') for i in range(len(gt_column))]
            weighted_scores = sum([f1 * weights[label] for label, f1 in zip(gt_column, f1_scores)])
            score += weighted_scores

        result = score / len(train_loader)
        average_loss = sum(loss_arr) / len(loss_arr)
        tr_loss.append(average_loss)
        tr_score.append(result)
        logging.info(f'Train | Epoch {epoch+1}/{epochs}, Loss: {average_loss}, Score: {result}')
        print(f'Train | Epoch {epoch+1}/{epochs}, Loss: {average_loss}, Score: {result}')
        
        scheduler.step()

        if best_score < result:
            best_model_wts = model.state_dict().copy()
            best_score = result
            save_model(model, lr_num, hidden_size,args.root, args= args)

    model.load_state_dict(best_model_wts)
    epoch_lst = list(range(0, epochs))
    logging.info(f'Best score {best_score}')

    vstr_loss.append(tr_loss)
    avg_loss.append(vl_loss)
    avg_tr_score.append(tr_score)
    avg_vl_score.append(vl_score)
    
    folder_path = f'./{args.save_path}/{lr_num}_final_{args.name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print(f"Best score {best_score}")
    logging.info(f"Best score {best_score}")
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_lst, tr_loss, label='Full Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Full Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_lst, tr_score, label='Full Training Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Full Training Score')
    plt.legend()

    plt.tight_layout()

    plt.savefig(f'{folder_path}/full_train.png')
    plt.close()
    
def cross_val(args):
    kf = KFold(n_splits=args.knum, shuffle=True, random_state=42)
    hidden_size = args.hidden
    num_layers = args.num_layers
    output_size = 7 
    input_size = args.inputs_size
    epochs = args.epochs
    learning_rate = args.lr

    # if not os.path.exists(os.path.join(args.train_ts_data_root, 'merged')):
    #     ts_train_data = get_ts_features(args.train_data_root)
    # else:
    #     ts_train_data = os.path.join(args.train_ts_data_root, 'merged')

    ts_train_data = get_ts_features(args.train_data_root)
    
    train_df = get_data(ts_train_data)
    train_label = load_label_data(args.label_path)

    train_x, train_y = merge_data(train_df, train_label)

    if args.duplicate:
        train_x, train_y = duplicate_labels(train_x, train_y)
        
    if args.transforms:
        train_transforms = ComposeTransforms([noise_transform, time_shift])
    else:
        train_transforms = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avg_loss = []
    vstr_loss = []
    avg_tr_score = []
    avg_vl_score = []
    avg_diff_score = []
    
    folder_path = f'{args.save_path}/{learning_rate}_{hidden_size}_{args.epochs}_{args.name}'
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(folder_path)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_x)):
        print(f'Fold {fold + 1}')
        X_train = [train_x[i] for i in train_idx]
        y_train = [train_y[i] for i in train_idx]
        X_val = [train_x[i] for i in val_idx]
        y_val = [train_y[i] for i in val_idx]

        train_dataset = CustomDataset(X_train, y_train, transform=train_transforms)
        val_dataset = CustomDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = 1, shuffle=False)

        model = args.model(input_size, hidden_size , num_layers , output_size)
        model.to(device)
        initialize_weights(model)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= args.decay)
        
        # scheduler 설정
        if args.scheduler:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs, eta_min= learning_rate * args.gamma)
        else:
            scheduler = MultiStepLR(optimizer, milestones=args.step, gamma=args.gamma)

        avg_train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, learning_rate, hidden_size, num_layers, fold_num=fold+1, avg_loss = avg_loss, vstr_loss= vstr_loss, avg_tr_score = avg_tr_score, avg_vl_score= avg_vl_score, avg_diff_score = avg_diff_score, args = args, folder_path = folder_path)
    
    average_losses = np.mean(avg_loss, axis=0)
    avg_tr_losses = np.mean(vstr_loss , axis=0)
    average_tr_score = np.mean(avg_tr_score, axis = 0)
    average_val_score = np.mean(avg_vl_score, axis = 0)

    plt.figure(figsize=(14, 6))
    plt.subplot(2,2,1)
    plt.plot(average_losses, label='Vaildation Loss')
    plt.plot(avg_tr_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training Loss Across Folds')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(average_tr_score, label='Training Score')
    plt.plot(average_val_score, label='Validation Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training and Validation Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{folder_path}/K-Fold_average.png')
    plt.close()
    
def train(args):
    if args.transforms:
        train_transforms = ComposeTransforms([noise_transform, time_shift])
    else:
        train_transforms = None
 
    # if not os.path.exists(os.path.join(args.train_ts_data_root, 'merged')):
    #     ts_train_data = get_ts_features(args.train_data_root)
    # else:
    #     ts_train_data = os.path.join(args.train_ts_data_root, 'merged')

    ts_train_data = get_ts_features(args.train_data_root)
    
    train_df = get_data(ts_train_data)
    train_label = load_label_data(args.label_path)
    
    train_x, train_y = merge_data(train_df, train_label)

    # Class 불균형 해소
    if args.duplicate:
        train_x, train_y = duplicate_labels(train_x, train_y)
        
    train_dataset = CustomDataset(train_x, train_y, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_size = args.hidden
    num_layers = args.num_layers
    output_size = 7 
    input_size = args.inputs_size
    avg_loss = []
    vstr_loss = []
    avg_tr_score = []
    avg_vl_score = []
    model = args.model(input_size, hidden_size , num_layers , output_size)
    model.to(device)
    torch.manual_seed(42)
    initialize_weights(model)

    learning_rate = args.lr
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= args.decay)
    
    if args.scheduler:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max= args.epochs, eta_min= learning_rate * args.gamma)
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.step, gamma=args.gamma)

    epochs = args.epochs
    
    # 평균, 분산 계산 , batch size 1로 설정
    # var_lst = cal_mean_var(train_loader)
    # print(var_lst)
        
    # 학습
    full_train(model, train_dataset, train_loader, criterion, optimizer, scheduler, epochs, learning_rate, hidden_size, num_layers, avg_loss, vstr_loss, avg_tr_score, avg_vl_score, args = args)
