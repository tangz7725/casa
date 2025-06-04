# import difflib
# import re
# from Bio import pairwise2
# from Bio.pairwise2 import format_alignment
# from fuzzywuzzy import fuzz
# import depthcharge.masses
# from depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder
# from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
#
# # 用列表存储每个标题基因组数据的结果
# def read_genomic_file(file_path):
#     # 用来存储结果的列表
#     sequences = []
#     # 用来存储当前序列的变量
#     current_sequence = ''
#     # 打开文件并逐行读取
#     with open(file_path, 'r') as file:
#         for line in file:
#             # 移除行尾的换行符
#             line = line.rstrip()
#             # 检查是否是序列的标题行
#             if line.startswith('>'):
#                 # 如果当前序列不为空，先将其加入结果列表
#                 if current_sequence:
#                     sequences.append(current_sequence)
#                     current_sequence = ''  # 重置当前序列存储
#             else:
#                 # 将序列行添加到当前序列
#                 current_sequence += line
#
#         # 将最后一个序列加入结果列表（如果存在）
#         if current_sequence:
#             sequences.append(current_sequence)
#
#     return sequences
#
# # 用换行符分割基因组每个标题的序列
# def process_genomic_file(input_file):
#     with open(input_file, 'r') as file:
#         sequences = []
#         sequence = []
#         for line in file:
#             line = line.strip()  # 去除每行的前后空白字符
#             if line.startswith('>'):  # 如果是标题行
#                 if sequence:
#                     # 将序列拼接成一行，并添加到结果列表中
#                     sequences.append(''.join(sequence))
#                     sequence = []  # 重置序列列表
#             else:
#                 sequence.append(line)  # 添加序列行到列表中
#         # 处理最后一个序列（如果有）
#         if sequence:
#             sequences.append(''.join(sequence))
#
#         # 将所有序列合并成一个字符串，序列之间用换行符分隔
#         return '\n'.join(sequences)
#
# # 模糊匹配，包括待查找序列query以及整个库（基因组序列）sequences
# def find_closest_match(sequences, query):
#     # 创建一个字符串列表，其中包含序列库中的所有序列
#     sequence_list = sequences.split('\n')
#
#     # 定义一个函数来计算序列与查询序列之间的相似度
#     def similar(s1, s2):
#         return difflib.SequenceMatcher(None, s1, s2).ratio()
#
#     # 找到最匹配的序列及其相似度
#     best_match = None
#     highest_score = 0
#     for seq in sequence_list:
#         score = similar(seq, query)
#         if score > highest_score:
#             highest_score = score
#             best_match = seq
#
#     return best_match, highest_score
#
#
# def fuzzywuzzy(sequences, query):
#     # 使用fuzzywuzzy计算两个字符串的模糊匹配程度
#     # ratio = fuzz.ratio(sequences, query)
#     partial_ratio = fuzz.partial_ratio(sequences, query)
#     return partial_ratio
#
#
# def re_search(sequences, query):
#     match = re.search(query, sequences)
#     return match
#
#
# # 注意这里的sequences是列表
# def Smith_Waterman(sequences, origin_query):
#     if not origin_query:
#         print(f"Normalized Score: {0:.5f}")
#         return origin_query
#
#     # 对序列的修饰清洗
#     query, modifications = extract_modifications(origin_query)
#
#     # 去除开头的$符号
#     query = remove_dollar(query)
#
#     # 计算总质量(此处没有考虑脱水缩合)
#     total_mass = calculate_total_mass(query)
#
#     # 搜索最佳匹配
#     best_alignment = None
#     best_score = 0
#     # best_normalized_score = 0
#     best_index = -1  # 用于记录最佳比对的序列索引，从0开始
#
#     # 定义评分标准
#     match_score = 2  # 匹配正确的得分
#     mismatch_penalty = -1  # 错配扣分
#     gap_open_penalty = -5  # 插入扣分
#     gap_extend_penalty = -5  # 增添扣分
#
#     # 计算最大可能得分（假设两个序列完全匹配）
#     max_score = len(query) * match_score
#
#     # # 计算最小可能得分（假设两个序列完全不同）
#     # min_score = 0
#
#     for index, sequence in enumerate(sequences):
#         alignments = pairwise2.align.localms(query, sequence, match_score, mismatch_penalty, gap_open_penalty, gap_extend_penalty)
#         # sorted_alignments = sorted(alignments, key=lambda x: x[2], reverse=True)
#         for alignment in alignments:
#             # 有多个匹配结果，可以考虑打印出来看看什么情况
#             # print("匹配结果:")
#             # print(format_alignment(*alignment))
#             score = alignment[2]
#             # # 计算归一化得分
#             # normalized_score = (score - min_score) / (max_score - min_score)
#             if score > best_score:
#                 best_alignment = alignment
#                 best_score = score
#                 # best_normalized_score = normalized_score
#                 best_index = index
#
#     # 说明完全匹配
#     if best_score == max_score:
#         return origin_query
#
#     # 打印最佳匹配结果
#     if best_alignment:
#         # print(format_alignment(*best_alignment))
#         alignment_res = format_alignment(*best_alignment)
#
#         # 对最佳匹配结果alignment_res鉴定，看看是否有空格（间隙）
#         gap_num = find_gaps_from_alignment(alignment_res)
#
#         # 找query匹配的开头位置
#         match = re.search(r'\d+', alignment_res.strip().split('\n')[0].strip())
#         if match:
#             startA = int(match.group(0)) - 1 - gap_num  # 注意这里-1才是起始位置
#         print(alignment_res)
#         print("best_alignment.start:"+str(best_alignment.start))
#         print(best_alignment.end)
#         # print(f"Best alignment found in sequence at index: {best_index}")
#         # print(f"Score: {best_score}")
#         # print(f"Normalized Score: {best_normalized_score:.5f}")
#         correction_A = get_sequence_segment(sequences, best_index, best_alignment.start, startA, len(query))
#         # correction_A, correction_B = get_sequence_segment(sequences, best_index, best_alignment.start, best_alignment.end, len(query))
#         print("the sequenceA after revise:"+correction_A)
#         # 计算质量是否超过误差
#         total_mass_after_revise = calculate_total_mass(correction_A)
#         if abs(total_mass - total_mass_after_revise) > 1:
#             return origin_query
#
#         # 对纠正后的结果添加回修饰
#         corrected_seq_with_mods = reapply_modifications('$'+correction_A, modifications)
#         print("the sequence after revise and reapply:"+corrected_seq_with_mods)
#         return corrected_seq_with_mods
#     else:
#         print("No significant alignment found.")
#         return origin_query
#
#
# def get_sequence_segment(sequences, index, begin, query_start, query_length):
#     # 检查索引是否在列表范围内
#     if index < 0 or index >= len(sequences):
#         return "Index out of range."
#
#     # 获取指定索引的序列
#     sequence = sequences[index]
#
#     # 检查起始位置和长度是否有效
#     if begin < 0 or begin >= len(sequence):
#         return "Begin position out of range."
#     if query_length < 0 or begin + query_length > len(sequence):
#         return "Length out of range."
#
#     # 返回指定的序列段
#     return sequence[begin - query_start:begin - query_start + query_length]
#
#
# def remove_dollar(seq):
#     return seq.replace('$', '', 1)
#
#
# def extract_modifications(seq):
#     pattern = re.compile(r'([\w$])([+-]\d+\.\d+)')
#     modifications = []
#     # amino_acid_counts = {}  # 用于跟踪每种氨基酸出现的次数
#
#     for match in pattern.finditer(seq):
#         # 氨基酸出现的次数默认为1
#         aa_times = 0
#         # 第一个捕获组的文本
#         amino_acid = match.group(1)
#         # 第二个捕获组的文本
#         modification = match.group(2)
#         # 更新氨基酸出现的次数
#         for i in range(match.start(1), -1, -1):
#             if amino_acid == seq[i]:
#                 aa_times += 1
#         modifications.append((amino_acid, modification, aa_times))
#
#     # 定位结束后，移除修饰信息
#     pattern_modify = re.compile(r'[+-]\d+\.\d+')
#     clean_seq = re.sub(pattern_modify, '', seq)
#
#     return clean_seq, modifications
#
#
# def preprocess_sequence(seq, modifications):
#     clean_seq = re.sub(r'\+\d+\.\d+', '', seq)
#     return clean_seq, modifications
#
#
# def reapply_modifications(seq, modifications):
#     modified_seq = seq  # 初始时，矫正后的序列就是最终序列
#     for amino_acid, modification, aa_times in modifications:
#         if amino_acid == '$':  # 特殊处理序列开头的修饰
#             modified_seq = modified_seq[0] + modification + modified_seq[1:]
#         else:
#             # 找到第 aa_times 次出现的氨基酸的位置
#             count = 0
#             for i, char in enumerate(modified_seq):
#                 if char == amino_acid:
#                     count += 1
#                     if count == aa_times:
#                         # 在该氨基酸前面添加修饰
#                         modified_seq = modified_seq[:i+1] + modification + modified_seq[i+1:]
#                         break
#     return modified_seq
#
#
# def find_gaps_from_alignment(alignment):
#     # 分行,其中第二行就是中间的匹配信息行
#     match_line = alignment.split('\n')[1].strip()
#     # 去除开头的空格
#     gaps_counter = 0
#     for char in match_line:
#         if char == ' ':
#             gaps_counter += 1
#
#     return gaps_counter
#
#
# def calculate_total_mass(query):
#     # 此处没有考虑脱水缩合
#     peptide_mass_calculator.masses['C'] = 103.009184505
#     return sum(peptide_mass_calculator.masses[aa] for aa in query)
#
#
# residues = "canonical"
# peptide_mass_calculator = depthcharge.masses.PeptideMass(residues)
# # 读取基因组数据
# if __name__ == "__main__":
#
#     # 调用函数，并传入文件路径
#     file_path = '../gene/whh_Candidatus_Thiodiazotropha_endoloripes.faa'  # 替换为你的文件路径
#     # 得到基因组序列
#     # genomic_sequences = process_genomic_file(file_path)
#
#     # read_genomic_file得到的基因组序列是列表
#     genomic_sequences = read_genomic_file(file_path)
#     print(genomic_sequences)
#     # 待搜索的序列 VEQKRLLKRG
#     # query_sequence = "$+17.946VE-15.995QKRLLK+0.984RG"
#     query_sequence = "$SSFGIQSALMLHM"  # 正确序列:HLSEGDAVK
#     # query_sequence = "$VEQKRLLKRG"
#     # YGPHTM+15.995AGDDPTK
#     peptide_after_correct = Smith_Waterman(genomic_sequences, query_sequence)
#     print("peptide_after_correct:"+peptide_after_correct)
#     # match = re_search(genomic_sequences, query_sequence)
#     # if match:
#     #     print(f"Pattern found at index {match.start()}")
#     # else:
#     #     print("Pattern not found")
#     # print(fuzzywuzzy(sequences=genomic_sequences, query=query_sequence))
#     # match, score = find_closest_match(genomic_sequences, query_sequence)
#     # print(f"最匹配的序列: {match}")
#     # print(f"相似度: {score}")
#
#
#
# # def edit_distance(s1, s2):
# #     m, n = len(s1), len(s2)
# #     dp = [[0] * (n + 1) for _ in range(m + 1)]
# #
# #     for i in range(m + 1):
# #         dp[i][0] = i
# #     for j in range(n + 1):
# #         dp[0][j] = j
# #
# #     for i in range(1, m + 1):
# #         for j in range(1, n + 1):
# #             if s1[i - 1] == s2[j - 1]:
# #                 dp[i][j] = dp[i - 1][j - 1]
# #             else:
# #                 dp[i][j] = 1 + min(dp[i - 1][j],  # 删除
# #                                    dp[i][j - 1],  # 插入
# #                                    dp[i - 1][j - 1])  # 替换
# #     return dp[m][n]
# #
# #
# # def calculate_match_score(predicted_peptide, genome_sequence):
# #     max_length = len(predicted_peptide)
# #     best_score = 0
# #     best_distance = float('inf')
# #
# #     # 遍历基因组序列，寻找最佳匹配
# #     for i in range(len(genome_sequence) - len(predicted_peptide) + 1):
# #         subseq = genome_sequence[i:i + len(predicted_peptide)]
# #         distance = edit_distance(predicted_peptide, subseq)
# #         current_score = (1 - distance / max_length) * 100
# #         best_score = max(best_score, current_score)
# #         best_distance = min(best_distance, distance)
# #
# #     return 1 - best_score/100
# #
# def count_ms_spectra(mgf_file_path):
#     spec_count = 0
#     reading_spectrum = False
#
#     with open(mgf_file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line == "BEGIN IONS":
#                 reading_spectrum = True
#                 spec_count += 1
#             elif line == "END IONS":
#                 reading_spectrum = False
#
#     return spec_count
#
#
# # if __name__ == "__main__":
# #     # # 示例
# #     # predicted_peptide = "PQVNGVVRTLNKTITILEQW"
# #     # genome_sequence = "RIAILTDAWYPQVNGVVRTLNKTITILEQWGHEILCINPELFRTLPMPTYPDIPLSLFPYGKIKKLLNDFKP"
# #     # score = calculate_match_score(predicted_peptide, genome_sequence)
# #     # print(f"Match score: {score:.2f}")
# #
# #     # 调用函数
# #     # 18247
# #     mgf_file_path = 'D:\postgraduate\casanova\dataset\clambacteria.mgf'  # 替换为你的MGF文件路径
# #
# #     # 5258
# #     mgf_file_path = '../sample_data/Candidatus02_big.mgf'  # 替换为你的MGF文件路径
# #     number_of_spectra = count_ms_spectra(mgf_file_path)
# #     print(f"Number of spectra in the MGF file: {number_of_spectra}")
#



def count_character_in_file(file_path, character):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            count = content.count(character)
            print(f"字符 '{character}' 在文件中出现了 {count} 次。")
    except FileNotFoundError:
        print("文件未找到，请检查路径是否正确。")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

# 使用示例
file_path = '../gene/Bacillus.faa'  # 这里替换成你的文件路径
character = '>'  # 这里替换成你想要统计的字符
count_character_in_file(file_path, character)