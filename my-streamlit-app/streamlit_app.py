import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
from datetime import datetime

st.set_page_config(page_title="交易分析报告生成器", layout="wide")
st.title("📈 自动化交易分析报告生成器")

uploaded_files = st.file_uploader("请上传一个或多个 Rithmic / ATAS 导出的 CSV 文件：", type="csv", accept_multiple_files=True)

if uploaded_files:
    @st.cache_data
    def load_and_clean_data(files):
        def extract_completed_orders(file):
            lines = file.getvalue().decode('utf-8').splitlines()
            start_index = None
            for i, line in enumerate(lines):
                if 'Completed Orders' in line:
                    start_index = i + 1
                    break
            if start_index is None:
                return pd.DataFrame()
            header = lines[start_index].replace('"', '').split(',')
            data = '\n'.join(lines[start_index + 1:])
            df = pd.read_csv(io.StringIO(data), names=header)
            return df

        dfs = [extract_completed_orders(f) for f in files]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all[df_all['Status'] == 'Filled']
        df_all = df_all[[
            'Account', 'Buy/Sell', 'Symbol', 'Avg Fill Price', 'Qty To Fill',
            'Update Time (CST)', 'Commission Fill Rate', 'Closed Profit/Loss']]
        df_all.columns = ['账户', '方向', '品种', '价格', '数量', '时间', '手续费', '盈亏']
        df_all['时间'] = pd.to_datetime(df_all['时间'], errors='coerce')
        df_all['方向'] = df_all['方向'].map({'B': 'Buy', 'S': 'Sell'})
        df_all['价格'] = pd.to_numeric(df_all['价格'], errors='coerce')
        df_all['数量'] = pd.to_numeric(df_all['数量'], errors='coerce')
        df_all['手续费'] = pd.to_numeric(df_all['手续费'], errors='coerce').fillna(0)
        df_all['盈亏'] = pd.to_numeric(df_all['盈亏'], errors='coerce')
        df_all = df_all.dropna(subset=['时间', '价格', '方向'])
        df_all = df_all.sort_values('时间').reset_index(drop=True)
        return df_all

    df_trades = load_and_clean_data(uploaded_files)
    st.success(f"已加载 {len(df_trades)} 条成交记录")

    # 品种过滤器
    all_symbols = sorted(df_trades['品种'].unique())
    selected_symbols = st.multiselect("选择要分析的品种（可多选）:", all_symbols, default=all_symbols)
    df_trades = df_trades[df_trades['品种'].isin(selected_symbols)]

    # 开平仓配对
    positions, records = [], []
    for _, row in df_trades.iterrows():
        if not positions:
            positions.append(row)
        else:
            last = positions[-1]
            if row['方向'] != last['方向'] and row['数量'] == last['数量'] and row['品种'] == last['品种'] and row['账户'] == last['账户']:
                entry = last
                exit = row
                pnl = (exit['价格'] - entry['价格']) * entry['数量'] if entry['方向'] == 'Buy' else (entry['价格'] - exit['价格']) * entry['数量']
                pnl -= (entry['手续费'] + exit['手续费'])
                records.append({
                    '账户': entry['账户'], '品种': entry['品种'],
                    '开仓时间': entry['时间'], '平仓时间': exit['时间'],
                    '方向': entry['方向'], '数量': entry['数量'],
                    '买入价': entry['价格'] if entry['方向'] == 'Buy' else exit['价格'],
                    '卖出价': exit['价格'] if entry['方向'] == 'Buy' else entry['价格'],
                    '盈亏': pnl
                })
                positions.pop()
            else:
                positions.append(row)

    df_pnl = pd.DataFrame(records)
    df_pnl['累计盈亏'] = df_pnl['盈亏'].cumsum()
    df_pnl['月份'] = df_pnl['平仓时间'].dt.to_period('M')

    # 统计指标
    sharpe = df_pnl['盈亏'].mean() / df_pnl['盈亏'].std() * np.sqrt(252)
    winrate = (df_pnl['盈亏'] > 0).mean()
    profit_ratio = df_pnl[df_pnl['盈亏'] > 0]['盈亏'].mean() / abs(df_pnl[df_pnl['盈亏'] < 0]['盈亏'].mean())
    cumulative = df_pnl['累计盈亏']
    max_drawdown = (cumulative - cumulative.cummax()).min()
    results = df_pnl['盈亏'].apply(lambda x: 1 if x > 0 else -1)
    streaks = results.ne(results.shift()).cumsum()
    max_win_streak = results[results > 0].groupby(streaks).size().max()
    max_loss_streak = results[results < 0].groupby(streaks).size().max()

    # 图表
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_pnl['平仓时间'], df_pnl['累计盈亏'], marker='o')
    ax1.set_title("累计盈亏曲线")
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    daily = df_pnl.groupby(df_pnl['平仓时间'].dt.date)['盈亏'].sum()
    ax2.plot(daily.index, daily.values, marker='o')
    ax2.set_title("每日盈亏波动")
    ax2.grid(True)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    hourly = df_pnl.groupby(df_pnl['平仓时间'].dt.hour)['盈亏'].mean()
    sns.barplot(x=hourly.index, y=hourly.values, ax=ax3, palette='coolwarm')
    ax3.set_title("每小时平均盈亏")
    ax3.grid(True)

    # 下载 Excel 报告
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_trades.to_excel(writer, sheet_name="原始交易", index=False)
        df_pnl.to_excel(writer, sheet_name="配对盈亏", index=False)
        df_pnl.groupby('账户')['盈亏'].agg(['sum', 'count', 'mean']).to_excel(writer, sheet_name="账户统计")
        df_pnl.groupby('月份')['盈亏'].agg(['sum', 'count', 'mean']).to_excel(writer, sheet_name="月度统计")
        df_pnl.groupby('品种')['盈亏'].agg(['sum', 'count', 'mean']).to_excel(writer, sheet_name="品种统计")
        summary = pd.DataFrame({
            '夏普比率': [sharpe],
            '胜率': [winrate],
            '盈亏比': [profit_ratio],
            '最大回撤': [max_drawdown],
            '最长连续盈利': [max_win_streak],
            '最长连续亏损': [max_loss_streak],
        })
        summary.to_excel(writer, sheet_name="统计指标", index=False)
        workbook = writer.book
        ws = workbook.add_worksheet("图表")
        writer.sheets['图表'] = ws
        for idx, fig in enumerate([fig1, fig2, fig3]):
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            ws.insert_image(f'A{idx * 20 + 1}', f'图{idx+1}.png', {'image_data': buf})

    st.download_button("📥 下载完整 Excel 报告", data=output.getvalue(), file_name="交易报告.xlsx")

    st.subheader("📄 配对交易盈亏明细")
    st.dataframe(df_pnl)
    st.subheader("📈 盈利曲线")
    st.pyplot(fig1)
    st.subheader("📊 每日盈亏")
    st.pyplot(fig2)
    st.subheader("🕒 每小时盈亏")
    st.pyplot(fig3)
    st.subheader("📌 核心统计")
    st.markdown(f"**夏普比率：** {sharpe:.2f}")
    st.markdown(f"**胜率：** {winrate:.2%}")
    st.markdown(f"**盈亏比：** {profit_ratio:.2f}")
    st.markdown(f"**最大回撤：** {max_drawdown:.2f}")
    st.markdown(f"**最长连续盈利笔数：** {max_win_streak}")
    st.markdown(f"**最长连续亏损笔数：** {max_loss_streak}")
