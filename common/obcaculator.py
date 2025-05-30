import re
from collections import defaultdict
import sys # Import sys to allow exiting

# 常用元素的原子量 (g/mol)
ATOMIC_WEIGHTS = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085,
    'P': 30.974, 'S': 32.06, 'Cl': 35.45, 'K': 39.098,
    'Ca': 40.078, 'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38,
    # --- 新增元素 ---
    'B': 10.81,   # 硼
    'F': 18.998,  # 氟
    'Ti': 47.867, # 钛
    'Mn': 54.938, # 锰
    'Br': 79.904, # 溴
    # 可以根据需要添加更多元素
}


def parse_formula(formula: str) -> dict:
    """
    解析化学式字符串，返回包含元素及其计数的字典。
    会捕获并返回 ValueError 以便在上层处理。
    """
    pattern = re.compile(r'([A-Z][a-z]*)(\d*)')
    matches = pattern.findall(formula)
    if not matches and formula.strip(): # 处理非空但无法匹配的情况
        raise ValueError(f"无法解析化学式 '{formula}'。请确保格式正确 (例如, C6H14O6, KNO3)。")
    if not formula.strip(): # 处理空字符串
        raise ValueError("化学式不能为空。")

    atom_counts = defaultdict(int)
    parsed_elements_string = "" # 用于验证解析是否覆盖整个字符串
    for element, count_str in matches:
        if element not in ATOMIC_WEIGHTS:
            raise ValueError(f"错误: 未知的元素 '{element}' 在化学式 '{formula}' 中。请将其添加到 ATOMIC_WEIGHTS 字典。")
        count = int(count_str) if count_str else 1
        atom_counts[element] += count
        parsed_elements_string += element + count_str

    # 检查是否有未解析的部分 (简单的完整性检查)
    if parsed_elements_string != formula.replace(" ", ""): # 忽略空格比较
         # 更复杂的检查可能需要，但这可以捕获一些基本错误
        pass # 允许部分匹配，但可以加一个警告如果需要更严格的检查

    if not atom_counts: # 如果循环后字典仍为空
        raise ValueError(f"无法从 '{formula}' 中提取任何元素。")

    return dict(atom_counts)

def calculate_mw(atom_counts: dict) -> float:
    """根据原子计数计算分子量。"""
    mw = 0.0
    for element, count in atom_counts.items():
        mw += ATOMIC_WEIGHTS[element] * count
    return mw

def calculate_ob_percent(atom_counts: dict, mw: float) -> float:
    """
    计算化合物的氧平衡 (OB%)。
    基于 C -> CO2, H -> H2O 的完全氧化假设。
    """
    if mw == 0:
        return 0.0 # 避免除以零

    x = atom_counts.get('C', 0)
    y = atom_counts.get('H', 0)
    z = atom_counts.get('O', 0)

    # 应用氧平衡公式 OB% = (z - 2*x - y/2) * 16 / mw * 100
    ob_percent = (z - 2 * x - y / 2) * 15.999 / mw * 100
    return ob_percent

def get_user_input():
    """交互式获取用户输入的组分化学式和比例信息。"""
    # component_data_input 将存储 (formula, proportion) 元组
    component_data_input = []

    while True:
        try:
            num_components = int(input("请输入混合物中的组分数量: "))
            if num_components > 0:
                break
            else:
                print("组分数量必须大于 0。")
        except ValueError:
            print("无效输入，请输入一个整数。")

    print("-" * 30)
    formulas_seen = set() # 用于检查重复的化学式输入

    for i in range(num_components):
        print(f"--- 输入组分 #{i+1} ---")
        while True:
            formula = input(f"请输入组分 #{i+1} 的化学式 (例如, C6H14O6, KNO3): ").strip()
            if not formula:
                print("化学式不能为空。")
                continue
            # 检查化学式是否已输入
            if formula in formulas_seen:
                 print(f"警告: 化学式 '{formula}' 已经输入过。")
                 # 你可以选择是阻止重复输入，还是允许但给出警告
                 # 这里我们允许重复，但在后续计算中它们会被视为同一物质的不同部分（如果需要分别处理比例，可能需要更复杂的逻辑）
                 # 如果想阻止，可以取消下面的注释并添加 continue
                 # print("请输入一个唯一的化学式，或将它们的比例合并后输入。")
                 # continue

            try:
                # 在这里尝试解析以尽早发现错误
                parse_formula(formula)
                # formulas_seen.add(formula) # 如果允许重复，可以在这里添加
                break # 化学式有效，跳出循环
            except ValueError as e:
                print(f"化学式错误: {e}")
                print("请重新输入。")

        while True:
            proportion_str = input(f"请输入化学式 '{formula}' 的质量比例 (例如, 70 或 30): ").strip()
            try:
                proportion = float(proportion_str)
                if proportion >= 0:
                    component_data_input.append((formula, proportion))
                    formulas_seen.add(formula) # 添加到已见集合
                    break
                else:
                    print("质量比例不能为负数。")
            except ValueError:
                print("无效输入，请输入一个数字。")
        # print("-" * 10) # 每个组分输入后的小分隔符 (可选)

    return component_data_input



def calculate_and_display_results(component_data_input: list):
    """
    执行计算并显示结果。
    现在接收一个包含 (formula, proportion) 元组的列表。
    """
    print("\n" + "="*15 + " 计算结果 " + "="*15)

    # --- 单个组分计算 ---
    # component_details 存储每个条目计算后的详细信息
    component_details = []
    print("--- 单个组分计算结果 ---")
    calculation_successful = True
    total_proportion = 0.0

    # 用于合并相同化学式的计算结果
    aggregated_data = {} # key: formula, value: {'mw': mw, 'ob_percent': ob, 'total_proportion': prop_sum}

    for formula, proportion in component_data_input:
        try:
            # 如果之前计算过这个化学式，直接复用结果
            if formula in aggregated_data:
                mw = aggregated_data[formula]['mw']
                ob_percent = aggregated_data[formula]['ob_percent']
                aggregated_data[formula]['total_proportion'] += proportion
            else:
                # 否则，进行计算
                atom_counts = parse_formula(formula)
                mw = calculate_mw(atom_counts)
                ob_percent = calculate_ob_percent(atom_counts, mw)
                # 存储计算结果以备复用和最终显示
                aggregated_data[formula] = {
                    'mw': mw,
                    'ob_percent': ob_percent,
                    'total_proportion': proportion
                }
                # 首次遇到该化学式时打印其基本信息
                print(f"组分 (化学式): {formula}")
                print(f"  分子量 (Mw): {mw:.3f} g/mol")
                print(f"  氧平衡 (OB%): {ob_percent:+.2f}%")

            # 记录每个输入条目的细节，包括其原始比例和计算出的属性
            component_details.append({
                'formula': formula,
                'proportion': proportion,
                'mw': mw,
                'ob_percent': ob_percent
            })
            total_proportion += proportion

        except ValueError as e:
            print(f"处理化学式 '{formula}' 时出错: {e}")
            calculation_successful = False
            # 即使一个组分失败，也尝试继续处理其他组分

    print("-" * 30)

    if not calculation_successful:
        print("由于一个或多个组分的错误，无法完成混合物计算。")
        return

    if total_proportion <= 0:
        print("警告: 所有组分的总比例为 0 或负数。无法计算混合物百分比和氧平衡。")
        return

    # --- 混合物计算 ---
    print("--- 混合物计算结果 ---")
    mixture_ob_percent = 0.0
    print("实际质量百分比:")

    # 按化学式汇总显示百分比
    for formula, data in aggregated_data.items():
        mass_fraction = data['total_proportion'] / total_proportion
        actual_percentage = mass_fraction * 100
        print(f"  - {formula}: {actual_percentage:.2f}%")
        # 使用每个化学式的总质量分数和其OB%来计算混合物OB%
        mixture_ob_percent += mass_fraction * data['ob_percent']

    # # 或者，如果你想显示每个输入的条目及其百分比（即使化学式重复）
    # print("按输入条目计算的实际质量百分比:")
    # mixture_ob_percent_alt = 0.0
    # for detail in component_details:
    #     mass_fraction = detail['proportion'] / total_proportion
    #     actual_percentage = mass_fraction * 100
    #     print(f"  - {detail['formula']} (输入比例 {detail['proportion']}): {actual_percentage:.2f}%")
    #     mixture_ob_percent_alt += mass_fraction * detail['ob_percent']
    # # 注意：mixture_ob_percent 和 mixture_ob_percent_alt 结果应该是一样的

    print(f"\n混合物总氧平衡 (OB%): {mixture_ob_percent:+.2f}%")
    print("-" * 30)

    # ... (后面的提示和安全警告保持不变) ...

    if mixture_ob_percent > 0:
        print("提示: 混合物为正氧平衡，意味着有多余的氧可用于氧化其他物质或产生富氧气体。")
    elif mixture_ob_percent < 0:
        print("提示: 混合物为负氧平衡，意味着氧气不足以完全氧化所有可燃物，可能产生 CO、H₂ 或碳烟等不完全燃烧产物。")
    else:
        print("提示: 混合物为零氧平衡（或接近零），理论上氧气量刚好足够完全氧化可燃物。")

    print("\n重要安全警告:")
    print("涉及氧化剂和燃料的混合物可能具有爆炸性或易燃性。")
    print("这些计算仅为理论值，实际行为可能受多种因素影响。")
    print("请勿在没有专业知识、适当设备和安全措施的情况下尝试制备或测试这些混合物。")
    print("="*40)




# --- 主程序循环 ---
if __name__ == "__main__":
    print("欢迎使用混合物氧平衡计算器！")
    while True:
        try:
            # 调用修改后的 get_user_input
            component_data_list = get_user_input()
            # 调用修改后的 calculate_and_display_results
            calculate_and_display_results(component_data_list)
        except ValueError as e:
            print(f"\n发生错误: {e}")
            print("请检查您的输入。")
        except Exception as e: # 捕获其他意外错误
             print(f"\n发生意外错误: {e}")

        # ... (询问是否继续的部分保持不变) ...
        while True:
            another = input("\n是否要进行另一次计算? (y/n): ").lower().strip()
            if another == 'y' or another == 'yes':
                print("\n" + "="*50 + "\n") # 清晰地开始下一次计算
                break
            elif another == 'n' or another == 'no':
                print("感谢使用，程序退出。")
                sys.exit() # 退出程序
            else:
                print("请输入 'y' 或 'n'。")

