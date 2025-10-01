
import os
import copy
import random
import argparse
import torch
import tqdm

parser = argparse.ArgumentParser(description='Weight Editor')
parser.add_argument('input', type=str, help='Path to .pth file')
parser.add_argument('-o', '--output', type=str, default="edited_weights", help='Path of folder to output .pth files')
parser.add_argument('-E', type=float, default=0, help='A value to be replaced. Enter a value from -1.0 to 1.0.')
parser.add_argument('-N', type=int, default=2, help='Number of weights to be replaced in random mode.')
parser.add_argument('-M', type=int, default=2, help='Number of random mode executions.')
parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2, 3], help='Edit mode (1: Row major, 2: Column major ,3: Random)')
parser.add_argument('--log_path', type=str, default="weight_edited.log", help='Edit log path')
args = parser.parse_args()

if __name__ == '__main__':
    if abs(args.E) > 1.0: 
        raise AssertionError(f'E={args.E} is not between -1. and 1.')
    print('Load model from', args.input)
    model = torch.load(args.input)
    os.makedirs(args.output, exist_ok=True)
    
    log = open(args.log_path, 'w')
    header = "input,output,mode,weight,row,col\n"
    log.write(header)
    cnt = 0
    if args.mode <= 2:
        for k, v in tqdm.tqdm(model.items()):
            if "weight" in k:
                row_num, col_num, _, _ = v.shape
                # row major mode
                if args.mode == 1:
                    for row in range(row_num):
                        org_values = copy.deepcopy(v[row])
                        v[row, :] = args.E
                        output_path = os.path.join(args.output, f"{cnt}.pth")
                        torch.save(model, output_path)
                        s = f'{args.input},{output_path},{args.mode},{k},{row:03},000-{col_num:03}\n'
                        log.write(s)
                        cnt += 1
                        v[row] = org_values
                # column major mode
                elif args.mode == 2:
                    for col in range(col_num):
                        org_values = copy.deepcopy(v[:, col])
                        v[:, col] = args.E
                        output_path = os.path.join(args.output, f"{cnt}.pth")
                        torch.save(model, output_path)
                        s = f'{args.input},{output_path},{args.mode},{k},000-{row_num:03},{col:03}\n'
                        log.write(s)
                        cnt += 1
                        v[:, col] = org_values
    # random choice mode
    elif args.mode == 3:
        keys = [k for k in model.keys() if "weight" in k]
        for m in tqdm.tqdm(range(args.M)):
            target_keys = random.sample(keys, args.N)
            org_values = dict()
            for key in target_keys:
                org_values[key] = copy.deepcopy(model[key])
                model[key][:] = args.E
            output_path = os.path.join(args.output, f"{cnt}.pth")
            torch.save(model, output_path)
            s = f'{args.input},{output_path},{args.mode},{"-".join(target_keys)},,\n'
            log.write(s)
            for key in target_keys:
                model[key][:] = org_values[key]
            cnt += 1