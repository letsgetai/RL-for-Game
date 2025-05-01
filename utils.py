import json
from pathlib import Path
import tempfile
from datetime import datetime
import csv
from io import StringIO
from typing import List, Dict, Any

def read_jsonl_return_list(file_path):
    """
    读取一个 .jsonl 文件，并将其内容解析为 JSON 列表。

    :param file_path: .jsonl 文件的路径
    :return: 包含所有 JSON 对象的列表
    """
    json_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"[WARNING] 跳过无效JSON行")
                continue
            json_list.append(json_obj)
    return json_list

def save_variable_log(var, prefix, log_dir="saved_logs"):
    """
    将变量保存为可读的.log文件，文件名格式：`<前缀>_<时间戳>.log`
    
    参数:
        var: 要保存的变量（支持基本类型、dict、list等）
        prefix: 文件名前缀（如 "data" → 生成 "data_20231015_142030.log"）
        log_dir: 存放日志的目录（默认 "saved_logs"）
    """
    # 确保目录存在
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.log"
    file_path = Path(log_dir) / filename
    
    try:
        # 写入变量内容（JSON格式，可读性更好）
        with open(file_path, "w", encoding="utf-8") as f:
            if isinstance(var, (dict, list, tuple, str, int, float, bool)):
                json.dump(var, f, indent=4, ensure_ascii=False)  # JSON格式化
            else:
                f.write(str(var))  # 非JSON兼容类型，直接转字符串
        
        # 打印日志信息
        print(f"[SUCCESS] 变量已保存到: {file_path}")
        print(f"变量类型: {type(var)}")
        #print(f"变量内容:\n{var}")
    
    except Exception as e:
        print(f"[ERROR] 保存失败: {e}")
        raise


def json_list_save_to_jsonl(data_list, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data_list:
                json_str = json.dumps(item, ensure_ascii=False)
                file.write(json_str + '\n')
        print(f"数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存数据时出现错误: {e}")


def jsonl_to_csv(input_file, output_file):
    """
    将JSONL文件转换为CSV文件
    
    参数:
        input_file (str): 输入的.jsonl文件路径
        output_file (str): 输出的.csv文件路径
    """
    # 读取所有JSON对象并收集所有可能的键
    records = []
    all_keys = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                records.append(record)
                all_keys.update(record.keys())
            except json.JSONDecodeError as e:
                print(f"警告: 跳过无效JSON行: {line.strip()}。错误: {e}")
    
    if not records:
        print("错误: 没有找到有效的JSON数据")
        return
    
    # 将键排序（可选，使列顺序一致）
    sorted_keys = sorted(all_keys)
    
    # 写入CSV文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys)
        writer.writeheader()
        
        for record in records:
            # 确保每条记录都有所有列（缺失值设为空）
            row = {key: record.get(key, '') for key in sorted_keys}
            writer.writerow(row)
    
    print(f"转换完成! 已保存到 {output_file}")

def reorder_json(data, priority_keys):
    """
    将指定的键值放到JSON输出的最前面
    
    参数:
        data (dict): 原始字典数据
        priority_keys (list): 需要前置的键列表
    
    返回:
        str: 重新排序后的JSON字符串
    """
    if not isinstance(data, dict):
        raise ValueError("输入数据必须是字典类型")
    
    if not isinstance(priority_keys, list):
        raise ValueError("priority_keys必须是列表类型")
    
    # 创建新字典，先放入优先键
    ordered_data = {}
    remaining_keys = [k for k in data.keys() if k not in priority_keys]
    
    # 添加优先键（保持输入顺序）
    for key in priority_keys:
        if key in data:
            ordered_data[key] = data[key]
    
    # 添加剩余键（保持原始顺序）
    for key in remaining_keys:
        ordered_data[key] = data[key]
    
    return ordered_data

def json_list_to_csv(json_list, output_file=None, encoding='utf-8'):
    """
    将包含多个JSON对象的列表转换为CSV文件或字符串
    
    参数:
        json_list (list): 包含多个字典/JSON对象的列表
        output_file (str, optional): 输出CSV文件路径。若为None则返回CSV字符串
        encoding (str): 文件编码格式，默认为utf-8
    
    返回:
        如果output_file为None，返回CSV字符串；否则返回None并直接写入文件
    """
    if not isinstance(json_list, list):
        raise ValueError("输入必须是列表类型")
    
    if not json_list:
        return "" if output_file is None else None

    # 收集所有可能的键（自动去重）
    all_keys = set()
    for item in json_list:
        if not isinstance(item, dict):
            raise ValueError(f"列表元素必须是字典类型，发现: {type(item)}")
        all_keys.update(item.keys())
    
    # 按首次出现的键顺序排序（可选：改为 sorted(all_keys) 按字母排序）
    key_order = []
    for item in json_list:
        for key in item.keys():
            if key not in key_order:
                key_order.append(key)
        if len(key_order) == len(all_keys):
            break
    
    # 处理输出目标
    if output_file is None:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=key_order)
        writer.writeheader()
        for item in json_list:
            writer.writerow({k: item.get(k, "") for k in key_order})
        return output.getvalue()
    else:
        with open(output_file, 'w', encoding=encoding, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=key_order)
            writer.writeheader()
            for item in json_list:
                writer.writerow({k: item.get(k, "") for k in key_order})
        return None


def deduplicate_json_list(data: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    """

    """
    if not isinstance(data, list):
        raise ValueError("输入必须是列表类型")
    if not isinstance(keys, list):
        raise ValueError("keys必须是列表类型")

    seen = set()
    result = []
    
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"列表元素必须是字典类型，发现: {type(item)}")
        
        # 创建去重键的组合（处理可能缺失的键）
        key_tuple = tuple(str(item.get(key, None)) for key in keys)
        
        if key_tuple not in seen:
            seen.add(key_tuple)
            result.append(item)
    
    return result



