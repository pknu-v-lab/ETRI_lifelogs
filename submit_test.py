from function import *
from dataset import *
from model import *
from preprocess import *
from torch.utils.data import DataLoader

def test(model, test_loader, device, args):
    model.eval() 
    model.to(device)
    all_data = []
    
    with torch.no_grad():
        for subject_id, date, inputs, length in test_loader: 
                
            inputs = inputs.to(device)
            outputs = model(inputs, length)

            outputs = (outputs >= args.threshold).int()
            
            output_data = outputs.tolist()
            all_data.append([subject_id, date[0], *output_data[0]])

    columns = ['subject_id', 'date', 'Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
    df = pd.DataFrame(all_data, columns=columns)
    df['subject_id'] = np.int32(df['subject_id'])

    df = df.sort_values(by=['subject_id', 'date'])

    save_path = f'{args.output_save_root}/output'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df.to_csv(f'{save_path}/Submit_{args.name}.csv', index=False)
    print("=======Test Complete=======")
            
def Submit_test_data(args):
    
    ts_test_data = get_ts_features(args.test_data_root)
    
    test_df = get_test(ts_test_data)
    test_x = merge_test(test_df)
    
    input_size = args.inputs_size 
    hidden_size = args.hidden
    num_layers =  args.num_layers 
    
    output_size = args.output_size
    model = Ensemble_Model(args.model, args.model_num, input_size, hidden_size, num_layers, output_size, args.weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = Test_Dataset(test_x)
    test_loader = DataLoader(test_dataset, shuffle = False)
    test(model, test_loader , device, args)


