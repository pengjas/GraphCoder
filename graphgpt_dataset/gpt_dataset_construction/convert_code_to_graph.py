import json
class top_module():
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.data = data
        self.get_inout_submodule()
        self.get_all_signals_of_nodes()
        self.get_connectivity()
        self.encode()
        self.constrcut_graph()

    def get_graph(self):
        return self.graph
    
    def process_src_dst(self):
        self.processed_src = [None] * len(self.src)
        self.processed_dst = [None] * len(self.dst)
        for i, src in enumerate(self.src):
            if isinstance(src, sub_module):
                self.processed_src[i] = src.instance_name
            else:
                self.processed_src[i] = src.symbol
        for i, dst in enumerate(self.dst):
            if isinstance(dst, sub_module):
                self.processed_dst[i] = dst.instance_name
            else:
                # print("dst:",dst)
                self.processed_dst[i] = dst.symbol
                
    def constrcut_graph(self):
        self.graph = {}
        self.graph['type'] = 'module_graph'
        self.graph['edge_attrs'] = []
        self.graph['has_edge_features'] = False
        self.graph['nodes'] = []
        self.graph['connectivity'] = [[], []]
        for in_p in self.encode_in:
            self.graph['nodes'].append({'id': self.encode_in[in_p], 'content': in_p, 'type': 'input port'})
        for out_p in self.encode_out:
            self.graph['nodes'].append({'id': self.encode_out[out_p], 'content': out_p, 'type': 'output port'})
        for module in self.encode_module:
            self.graph['nodes'].append({'id': self.encode_module[module], 'content': module, 'type': 'submodule'})
        self.process_src_dst()
        for src in self.processed_src:
            self.graph['connectivity'][0].append(self.encode_dict[src])
        for dst in self.processed_dst:
            self.graph['connectivity'][1].append(self.encode_dict[dst])

    def encode(self):
        num_in = len(self.in_port)
        num_out = len(self.out_port)
        num_submodules = len(self.submodules)
        # num_nodes = num_in + num_out + num_submodules
        encode_in = {}
        encode_out = {}
        encode_module = {}
        encode_dict = {}
        for i in range(num_in):
            encode_in[self.in_port[i]] = i
            encode_dict[self.in_port[i]] = i
        for i in range(num_out):
            encode_out[self.out_port[i]] = num_in + i
            encode_dict[self.out_port[i]] = num_in + i
        for i in range(num_submodules):
            encode_module[self.submodules[i].instance_name] = num_in + num_out + i
            encode_dict[self.submodules[i].instance_name] = num_in + num_out + i
        self.encode_in = encode_in
        self.encode_out = encode_out
        self.encode_module = encode_module
        self.encode_dict = encode_dict


    def get_inout_submodule(self):
        self.in_port = []
        self.out_port = []
        self.submodules = []
        members = self.data['design']['members'][1]['body']['members']
        self.members = members
        for member in members:
            if member['kind'] == 'Port' and member['direction'] == 'In':
                self.in_port.append(member['name'])   
            elif member['kind'] == 'Port' and member['direction'] == 'Out':
                self.out_port.append(member['name'])  
            elif member['kind'] == 'Instance':           
                self.submodules.append(sub_module(member))
    
    def get_all_signals_of_nodes(self):
        self.signals = {}
        for in_p in self.in_port:
            symbol = signals(in_p)
            self.signals[symbol] = [symbol]
        # for out_p in self.out_port:
        #     symbol = signals(out_p)

        #     self.signals[symbol] = symbol
        for module in self.submodules:
            for signal in module.internal_signals:
                # signal, has_existed = self.signal_has_existed(signal, self.signals)
                if signal not in self.signals:
                    self.signals[signal] = []
                # if signal not in self.signals:
                #     self.signals[signal] = []
                # elif isinstance(self.signals[signal], str):
                #     self.signals[signal] = []
                # print(self.signals[signal])
                # if not isinstance(self.signals[signal], list):
                #     self.signals[signal] = [self.signals[signal]]
                # print("+++++++++++++++++++")
                # print("signal:",signal.symbol)
                # print("module:",module.instance_name)
                if not isinstance(self.signals[signal], list):
                    self.signals[signal] = []
                # for key in self.signals:
                #     print(key.symbol)
                self.signals[signal].append(module)
            
        # for key in self.signals:
        #     print("-----------------------")
        #     print("key.symbol", key.symbol)
        #     for sig in self.signals[key]:
        #         print("related node")
        #         if isinstance(sig, sub_module):
        #             print(sig.instance_name)
        #         else:
        #             print(sig.symbol)
        # print("here")

    def signal_has_existed(self, signal, list):
        for key in list:
            if not isinstance(key, signals):
                continue
            if key.symbol == signal.symbol and key.select_type == signal.select_type and key.start == signal.start and key.end == signal.end:
                return key, True
        
        return signal, False


    def get_connectivity(self):
        self.src = []
        self.dst = []
        node_point_to_module = []
        for module in self.submodules:
            node_point_to_module = self.process_one_module(module)
            effective_node_point_to_module = []
            for node in node_point_to_module:
                is_related, keys = self.is_node_related(node)
                if is_related:
                    for key in keys:
                        # print("-----------------")
                        # print("key:",key)
                        # print("self.signals[key]:",self.signals[key])
                        tmp = self.signals[key]
                        if not isinstance(self.signals[key], list):
                            tmp = [self.signals[key]]
                        effective_node_point_to_module.extend(tmp)
                        # print("+++++++++++")
                    # if node in self.in_port or node in self.out_port:
                    #     effective_node_point_to_module.append(node)
                    # else:
                    #     effective_node_point_to_module.extend(self.signals[node])
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print("effective_node_point_to_module:",effective_node_point_to_module)
            effective_node_point_to_module = list(set(effective_node_point_to_module))
            self.src.extend(effective_node_point_to_module)
            dst_node = [module] * len(effective_node_point_to_module)
            self.dst.extend(dst_node)
        for out_p in self.out_port:
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print("out_p:",out_p)
            out_signal = signals(out_p)
            node_point_to_out = self.find_src_nodes(out_signal)
            # print("node_point_to_out:",node_point_to_out)
            # for node in node_point_to_out:
            #     print("node:",node.symbol)
            node_point_to_out.append(out_signal)
            effective_node_point_to_out = []
            # print("out_p:",out_p)
            # print("node_point_to_out:",node_point_to_out)
            for node in node_point_to_out:
                is_related, keys = self.is_node_related(node)
                if is_related:
                    # print("node:",node)
                    # print("self.signals[node]:",self.signals[node])
                    # if isinstance(self.signals[node], str):
                    #     effective_node_point_to_out.append(node)
                    # else:
                    #     effective_node_point_to_out.extend(self.signals[node])
                    for key in keys:
                        tmp = self.signals[key]
                        if not isinstance(self.signals[key], list):
                            tmp = [self.signals[key]]
                        effective_node_point_to_out.extend(tmp)
            # print("**************************")
            effective_node_point_to_out = list(set(effective_node_point_to_out))
            # print("effective_node_point_to_out:",effective_node_point_to_out)
            self.src.extend(effective_node_point_to_out)
            dst_node = [out_signal] * len(effective_node_point_to_out)
            self.dst.extend(dst_node)

        
    def is_node_related(self, node):
        is_related = False
        keys = []
        for key in self.signals:
            if key.symbol == node.symbol:
                if key.select_type == None or node.select_type == None:
                    is_related = True
                    keys.append(key)
                elif key.start <= node.end and node.start <= key.end:
                    is_related = True 
                    keys.append(key)

        return is_related, keys

    def process_one_module(self, module):
        node_point_to_module = []
        for in_p in module.in_port:
            # print("in_p:",in_p)
            # print("module",module.instance_name)
            dst = module.connections[in_p]
            # print("dst:",dst)
            # print("dst length:",len(dst))
            # print("dst")
            for d in dst:
                node_point_to_module.extend(self.find_src_nodes(d))
        node_point_to_module = list(set(node_point_to_module))
        # print("module:",module.instance_name)
        # for node in node_point_to_module:
        #     print("node:",node.symbol)
        #     print("****")
        return node_point_to_module
            

    def find_src_nodes(self, signal):
        # print('----------------------')
        # print("signal:",signal.symbol)
        if signal.symbol in self.in_port:
            return [signal] 
        all_related_nodes = []
        next_level_nodes = []
        for member in self.members:
            if member['kind'] == 'Instance':
                continue
            next_level_nodes.extend(self.process_one_member(member, signal))
        # print("++++++++++++++++++++++")
        # print("signal:",signal)
        # print("next_level_nodes" ,next_level_nodes)
        # for node in next_level_nodes:
            # print("node:",node.symbol)
            # print("****")
        all_related_nodes.extend(next_level_nodes)
        for node in next_level_nodes:
            all_related_nodes.extend(self.find_src_nodes(node))
        # all_related_nodes = list(set(all_related_nodes))
        if len(all_related_nodes) == 0:
            all_related_nodes.append(signal)
        # all_related_nodes = list(set(tuple(node) for node in all_related_nodes))
        # print('-----------------------')
        # print(all_related_nodes)
        all_related_nodes = list(set(all_related_nodes))
        
        return all_related_nodes


    def process_one_member(self, member, signal):
        expressions = dfs_search(member)
        # print("expressions:",expressions)
        related_nodes = []
        # print("oooooooooooooooooooo")
        # print("lengt of expressions:",len(expressions))
        for expression in expressions:
            if expression['kind'] == 'Assignment':
                left = expression['left']
                right = expression['right']
                left_symbols = collect_symbols(left)
                right_symbols = collect_symbols(right)
                right_symbols = list(set(right_symbols))
            elif 'initializer' in expression:
                left_symbols = [signals(expression['name'])]
                right_symbols = collect_symbols(expression['initializer'])
            # print("left_symbols:",left_symbols)
            # print("right_symbols:",right_symbols)
            # key, has_existed = self.signal_has_existed(signal, left_symbols)
            # print("ppppppppppppp")
            # for left in left_symbols:
                # print("$$$$$$$$$$")
                # print("symbol:",left.symbol)
            if self.is_signal_related(signal, left_symbols):

                related_nodes.extend(right_symbols)
        # print("related_nodes:",related_nodes)
        return related_nodes

    def is_signal_related(self, signal, symbols):
        is_related = False
        for symbol in symbols:
            if signal.symbol == symbol.symbol:
                if symbol.select_type == None or signal.select_type == None:
                    is_related = True
                elif symbol.start <= signal.end and signal.start <= symbol.end:
                    is_related = True
        return is_related

def collect_symbols(d):
    symbols = []  # 用于存储所有 'symbol' 的值

    # 检查当前字典是否为字典类型
    if isinstance(d, dict):
        if 'kind' in d and d['kind'] == 'ElementSelect':
            # print("-----------------")
            # print(d)
            symbol = get_formal_name(d['value']['symbol'])
            select_type = 'ElementSelect'
            start = d['selector']['value']
            end = d['selector']['value']
            symbols.append(signals(symbol, select_type, start, end))
            return symbols
        elif 'kind' in d and d['kind'] == 'RangeSelect':
            symbol = get_formal_name(d['value']['symbol'])
            select_type = 'RangeSelect'
            start = d['left']['value']
            end = d['right']['value']
            symbols.append(signals(symbol, select_type, start, end))
            return symbols
        # 如果字典中包含 'symbol' 键，收集其值
        else:
            if 'symbol' in d:
                symbol = get_formal_name(d['symbol'])
                symbols.append(signals(symbol))
                # symbols.append(get_formal_name(d['symbol']))

            # 遍历字典的每个键值对
            for key, value in d.items():
                symbols.extend(collect_symbols(value))  # 递归调用并合并结果

    # 如果是列表，遍历每个元素
    elif isinstance(d, list):
        for item in d:
            symbols.extend(collect_symbols(item))  # 递归调用并合并结果

    return symbols  # 返回所有收集到的 'symbol' 值


def dfs_search(dictionary):
    results = []
    
    def recurse(d):
        if isinstance(d, dict):
            if d.get('kind') == 'Assignment':
                results.append(d)
            elif 'initializer' in d:
                results.append(d)
            for key, value in d.items():
                recurse(value)
        elif isinstance(d, list):
            for item in d:
                recurse(item)
    
    recurse(dictionary)
    return results

def get_formal_name(name):
    return name.split(' ')[1]


class sub_module():
    def __init__(self, instance):
        self.instance = instance
        self.instance_name = instance['name']
        self.module_name = instance['body']['name']
        self.get_inout_connections()
        self.get_internal_signals()

    def get_inout_connections(self):
        self.in_port = []
        self.out_port = []
        self.connections = {}
        members = self.instance['body']['members']
        connections = self.instance['connections']
        self.get_connections(connections)
        for member in members:
            if member['kind'] == 'Port' and member['direction'] == 'In':
                self.in_port.append(member['name'])
                # self.connections[member['name']] =  get_formal_name(member['internalSymbol'])  
            elif member['kind'] == 'Port' and member['direction'] == 'Out':
                self.out_port.append(member['name'])
                # self.connections[member['name']] =  get_formal_name(member['internalSymbol'])
    
    def get_connections(self, connections):
        for connection in connections:
            symbols = collect_symbols(connection['expr'])
            
            self.connections[connection['port']['name']] = symbols
            # if connection['expr']['kind'] == 'NamedValue':
            #     symbol = get_formal_name(connection['expr']['symbol'])
            #     self.connections[connection['port']['name']] = signals(symbol)
            # elif connection['expr']['kind'] == 'Assignment':
            #     left = connection['expr']['left']
            #     if left['kind'] == 'ElementSelect':
            #         symbol = get_formal_name(connection['expr']['left']['value']['symbol']) 
            #         select_type = 'ElementSelect'
            #         start = left['selector']['value']
            #         end = left['selector']['value']
            #         self.connections[connection['port']['name']] = signals(symbol, select_type, start, end) 
            #     else:
            #         symbol = get_formal_name(connection['expr']['left']['symbol'])
            #         self.connections[connection['port']['name']] = signals(symbol)
            # elif connection['expr']['kind'] == 'Conversion':
            #     # print(connection)
            #     if 'symbol' in connection['expr']['operand']:
            #         symbol = get_formal_name(connection['expr']['operand']['symbol'])
            #     elif connection['expr']['operand']['kind'] == 'Concatenation':
            #         symbol = []
            #         for operand in connection['expr']['operand']['operands']:
            #             if 'symbol' in operand:
            #                 symbol.append(signals(get_formal_name(operand['symbol'])))
            #     # symbol = get_formal_name(connection['expr']['operand']['symbol'])
            #     self.connections[connection['port']['name']] = symbol
            # elif connection['expr']['kind'] == 'ElementSelect':
            #     symbol = get_formal_name(connection['expr']['value']['symbol'])
            #     select_type = 'ElementSelect'
            #     start = connection['expr']['selector']['value']
            #     end = connection['expr']['selector']['value']
            #     self.connections[connection['port']['name']] = signals(symbol, select_type, start, end)
            # elif connection['expr']['kind'] == 'RangeSelect':
            #     symbol = get_formal_name(connection['expr']['value']['symbol'])
            #     select_type = 'RangeSelect'
            #     start = connection['expr']['left']['value']
            #     end = connection['expr']['right']['value']
            #     self.connections[connection['port']['name']] = signals(symbol, select_type, start, end)
    
    
    def get_internal_signals(self):
        self.internal_signals = []
        # for in_p in self.in_port:
        #     self.internal_signals.append(self.connections[in_p])
        for out_p in self.out_port:
            self.internal_signals.extend(self.connections[out_p])

class signals():
    def __init__(self, symbol: str, select_type=None, start=None, end=None):
        self.symbol = symbol
        self.select_type = select_type
        if start is not None:
            start = int(start)
            end = int(end)
            if start > end:
                start, end = end, start
        self.start = start
        self.end = end
        # print("start:",start)
        # print("end:",end)
        self.range = None
        if select_type is not None:

            self.range = list(range(start, end+1))
    def __hash__(self):
        return hash((self.symbol, self.select_type, self.start, self.end))
    def __eq__(self, other):
        return self.symbol == other.symbol and self.select_type == other.select_type and self.start == other.start and self.end == other.end


import json
import os

def jsonline_iter(file_path: str):
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)

def example_to_jsonline(examples: dict, save_file: str):
    with open(save_file, "a") as f:
            f.write(json.dumps(examples) + "\n")

if __name__ == "__main__":
    cnt = 0
    cnt_list = []
    for item in jsonline_iter("valid_multi_module_and_gpt4.jsonl"):
        instruction = item["Instruction"]
        if isinstance(item["Response"], list):
            code = item["Response"][0]
        else:
            code = item["Response"]
        if os.path.exists("test.v"):
            os.system("rm test.v")
        if os.path.exists(path="test.json"):
            os.system("rm test.json")
        with open("test.v", "w") as file:
            file.write(code)
        os.system("slang test.v -ast-json test.json")
        cnt = cnt + 1
        try:
            test = top_module('./test.json')
            graph = test.get_graph()
            # instruction = 'This is the graph to describe interconnections among modules.' + '\n'  + str(data) + '\n' + instruction
            dict_to_save = {
                "Instruction": instruction,
                "Response": code
            }
            example_to_jsonline(dict_to_save, "./conversations.jsonl")
            example_to_jsonline(graph, "./graph.jsonl")
        except:
            cnt_list.append(cnt)
            print("error:", cnt-1)
            
            continue

    print("error list:", cnt_list)