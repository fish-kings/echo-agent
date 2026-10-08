/*!
用于驱动有限次 Agent 调用的通用运行时基础组件。

| 模块 | 说明 |
|--------|-------------|
| [`turn_driver`] | 单次有限 Agent 回合：`AgentTurnDriver`、`TurnRequest`、`TurnOutcome`、`EventSink` |
*/

pub mod turn_driver;

pub use turn_driver::{
    AgentTurnDriver, EventSink, SinkControl, TurnDeliveryOutcome, TurnInput, TurnInputReceipt,
    TurnInputState, TurnMode, TurnOutcome, TurnReceipt, TurnRequest,
};

/// 重新导出回合运行时所使用的结果与传输类型。
pub use echo_core::agent::{EventEnvelope, EventIdentity, ExecutionUsage};
