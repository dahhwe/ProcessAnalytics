import logging

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.visualization.petri_net import visualizer as pn_visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)


def read_and_parse_xes(file_path):
    logging.info(f"Чтение XES файла из {file_path}")
    log = pm4py.read_xes(file_path)
    return log


def convert_log_to_dataframe(log):
    logging.info("Преобразование журнала в DataFrame")
    df = pm4py.convert_to_dataframe(log)
    return df


def preprocess_data(df):
    logging.info("Предварительная обработка данных")
    required_columns = ['concept:name', 'time:timestamp', 'org:resource', 'case:concept:name', 'org:role']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Отсутствует обязательный столбец: {col}")

    df.replace(['UNKNOWN', 'UNDEFINED'], np.nan, inplace=True)
    df.dropna(subset=['concept:name', 'time:timestamp'], inplace=True)

    df['org:resource'] = df['org:resource'].fillna('unknown')
    df['org:role'] = df['org:role'].fillna('unknown')
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
    df['concept:name'] = df['concept:name'].str.title()
    df['org:resource'] = df['org:resource'].str.title()
    df['org:role'] = df['org:role'].str.title()

    additional_columns = ['Permit TaskNumber', 'Permit ProjectNumber', 'Permit ActivityNumber']
    for col in additional_columns:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    df.drop_duplicates(inplace=True)

    if 'Amount' in df.columns:
        df = df[df['Amount'] >= 0]

    df.sort_values(by=['case:concept:name', 'time:timestamp'], inplace=True)

    case_sizes = df.groupby('case:concept:name').size()
    single_event_cases = case_sizes[case_sizes == 1].index
    df = df[~df['case:concept:name'].isin(single_event_cases)]

    return df


def construct_graph(log):
    logging.info("Построение графа сети Петри")
    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log)
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters={"format": "svg"})
    pn_visualizer.save(gviz, "petri_net_visualization.svg")
    logging.info("Граф сети Петри сохранён в файл 'petri_net_visualization.svg'")
    return net, initial_marking, final_marking


def analyze_graph(log, net, initial_marking, final_marking):
    logging.info("Анализ графа сети Петри")

    # Анализ отклонений
    aligned_traces = token_replay.apply(log, net, initial_marking, final_marking)
    deviations = [trace for trace in aligned_traces if not trace['trace_is_fit']]
    logging.info(f"Количество отклоняющихся трасс: {len(deviations)}")

    # Анализ продолжительности случаев
    case_durations = case_statistics.get_all_case_durations(log, parameters={
        case_statistics.Parameters.TIMESTAMP_KEY: 'time:timestamp'})
    mean_duration = sum(case_durations) / len(case_durations)
    std_duration = np.std(case_durations)
    threshold = mean_duration + std_duration
    late_cases = [case for case in case_durations if case > threshold]
    logging.info(f"Средняя продолжительность случая: {mean_duration}")
    logging.info(f"Стандартное отклонение продолжительности случая: {std_duration}")
    logging.info(f"Количество просроченных случаев: {len(late_cases)}")

    # Анализ узких мест
    resource_counts = attributes_filter.get_attribute_values(log, "org:resource")
    bottleneck = max(resource_counts, key=resource_counts.get)
    logging.info(f"Узкое место (ресурс с наибольшей нагрузкой): {bottleneck}")

    logging.info("Анализ успешно завершен")
    return len(deviations), len(late_cases), bottleneck


def analyze_cycles(log):
    logging.info("Анализ на наличие циклов")
    try:
        cyclic_traces = [
            trace for trace in log if
            len(set(event["concept:name"] for event in trace if "concept:name" in event)) < len(trace)
        ]
        logging.info(f"Количество циклов в трассах: {len(cyclic_traces)}")
        return len(cyclic_traces)
    except Exception as e:
        logging.error(f"Ошибка при анализе циклов: {e}")
        return 0


def main():
    logging.info("Выполнение практической работы предполагает решение следующих задач:")
    logging.info("1. Выполнить предварительную обработку данных")
    log = read_and_parse_xes('InternationalDeclarations.xes')
    df = convert_log_to_dataframe(log)
    df = preprocess_data(df)

    logging.info("2. Построить граф для выбранного бизнес-процесса")
    net, initial_marking, final_marking = construct_graph(log)
    logging.info("Построение графа завершено, переход к анализу")

    logging.info(
        "3. Провести анализ полученного графа на предмет наличия в нем узких мест, циклов, отклонений и оптимальных путей.")
    logging.info(
        "Определить и проанализировать те экземпляры процесса, выполнение которых длится намного дольше остальных")
    num_deviations, num_late_cases, bottleneck = analyze_graph(log, net, initial_marking, final_marking)
    num_cycles = analyze_cycles(log)

    logging.info(f"Количество отклоняющихся трасс: {num_deviations}")
    logging.info(f"Количество просроченных случаев: {num_late_cases}")
    logging.info(f"Количество циклов в трассах: {num_cycles}")
    logging.info(f"Узкое место в процессе: {bottleneck}")

    logging.info("4. Предложить варианты оптимизации выбранного вами бизнес-процесса.")
    logging.info("Предложение: перераспределить нагрузку с узкого места и устранить циклы в трассах.")


if __name__ == "__main__":
    main()
