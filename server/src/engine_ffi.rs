use serde::Serialize;
use std::fmt;

pub const FLINT_ENGINE_ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineStatus {
    Ok = 0,
    InvalidInput = 1,
    ContextOverflow = 2,
    Cancelled = 3,
    Internal = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineEventKind {
    Tokens = 0,
    Done = 1,
    Error = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct EngineTelemetry {
    pub prefill_tokens: u32,
    pub cached_prefix_tokens: u32,
    pub decode_tokens: u32,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_rounds: u32,
    pub spec_accepted: u32,
}

pub type Telemetry = EngineTelemetry;

#[repr(C)]
pub struct EngineEvent {
    pub req_id: u64,
    pub kind: EngineEventKind,
    pub status: EngineStatus,
    pub token_ids: *const i32,
    pub n_tokens: usize,
    pub telemetry: EngineTelemetry,
    pub error_message: *const std::ffi::c_char,
}

#[repr(C)]
pub struct RequestParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u64,
    pub stop_token_ids: *const i32,
    pub n_stop_token_ids: usize,
    pub ignore_eos: u8,
}

#[repr(C)]
pub struct MediaSlice {
    pub kind: u32,
    pub data: *const u8,
    pub size: usize,
    pub token_offset: u32,
}

#[derive(Default)]
pub struct Request<'a> {
    pub token_ids: &'a [i32],
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u64,
    pub stop_token_ids: &'a [i32],
    pub ignore_eos: bool,
}

#[derive(Debug)]
pub enum Event {
    Tokens(Vec<i32>),
    Done(Telemetry),
}

#[derive(Debug)]
pub struct EngineError {
    pub status: EngineStatus,
    pub message: String,
}

impl EngineError {
    fn new(status: EngineStatus, message: &str) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.status, self.message)
    }
}

impl std::error::Error for EngineError {}

pub use backend::Engine;

fn validate_config(config: &str) -> Result<(), EngineError> {
    if config.is_empty() || config.as_bytes().contains(&0) {
        return Err(EngineError::new(
            EngineStatus::InvalidInput,
            "expected nonempty config without NUL",
        ));
    }
    Ok(())
}

#[cfg(not(feature = "link-engine"))]
mod backend {
    use super::*;
    use std::cell::Cell;

    pub struct Engine {
        submitted: Cell<u64>,
        cancellations: Cell<u64>,
    }
    impl Engine {
        pub fn new(config: &str) -> Result<Self, EngineError> {
            validate_config(config)?;
            Ok(Self {
                submitted: Cell::new(0),
                cancellations: Cell::new(0),
            })
        }
        pub fn stats_json(&self) -> Result<String, EngineError> {
            Ok(serde_json::json!({
                "backend": "stub", "mode": "smoke", "completed_requests": self.submitted.get(),
                "active_requests": 0, "queued_requests": 0, "abi_version": FLINT_ENGINE_ABI_VERSION,
                "gpu_name": null, "sm_major": 0, "sm_minor": 0,
                "vram_total_bytes": 0, "vram_free_bytes": 0,
                "submitted_requests": self.submitted.get(), "cancelled_requests": self.cancellations.get()
            }).to_string())
        }
        pub fn submit(&self, req: &Request<'_>) -> Result<(u64, Vec<Event>), EngineError> {
            let id = self.submitted.get() + 1;
            self.submitted.set(id);
            if req.token_ids.is_empty() {
                return Err(EngineError::new(EngineStatus::InvalidInput, "empty prompt"));
            }
            let prefill_tokens =
                req.token_ids.len().try_into().map_err(|_| {
                    EngineError::new(EngineStatus::ContextOverflow, "prompt too long")
                })?;
            let emit = req.max_tokens > 0 && !req.stop_token_ids.contains(&42);
            let mut events = Vec::new();
            if emit {
                events.push(Event::Tokens(vec![42]));
            }
            events.push(Event::Done(Telemetry {
                prefill_tokens,
                decode_tokens: emit.into(),
                decode_rounds: emit.into(),
                ..Default::default()
            }));
            Ok((id, events))
        }
        pub fn cancel(&self, id: u64) {
            if id > 0 && id <= self.submitted.get() {
                self.cancellations.set(self.cancellations.get() + 1);
            }
        }
    }
}

#[cfg(feature = "link-engine")]
mod backend {
    use super::*;
    use std::{
        ffi::{CStr, CString, c_char, c_void},
        panic::{AssertUnwindSafe, catch_unwind},
        ptr::{self, NonNull},
        slice,
        sync::mpsc,
    };

    #[repr(C)]
    struct EngineHandle {
        _private: [u8; 0],
    }
    type TokenCallback = unsafe extern "C" fn(*const EngineEvent, *mut c_void);
    unsafe extern "C" {
        fn engine_abi_version() -> u32;
        fn engine_create(
            config_json: *const c_char,
            err_out: *mut *mut c_char,
        ) -> *mut EngineHandle;
        fn engine_destroy(h: *mut EngineHandle);
        fn engine_submit(
            h: *mut EngineHandle,
            params: *const RequestParams,
            token_ids: *const i32,
            n_tokens: usize,
            media: *const MediaSlice,
            n_media: usize,
            cb: TokenCallback,
            user_data: *mut c_void,
        ) -> u64;
        fn engine_cancel(h: *mut EngineHandle, id: u64);
        fn engine_stats_json(h: *mut EngineHandle) -> *mut c_char;
        fn engine_string_free(s: *mut c_char);
    }

    // Both owned-string APIs use the engine allocator.
    unsafe fn take_string(ptr: *mut c_char) -> Option<String> {
        let ptr = NonNull::new(ptr)?;
        let s = unsafe { CStr::from_ptr(ptr.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        unsafe { engine_string_free(ptr.as_ptr()) };
        Some(s)
    }

    type CallbackState = mpsc::Sender<Result<Event, EngineError>>;

    unsafe extern "C" fn collect(event: *const EngineEvent, data: *mut c_void) {
        let event = unsafe { &*event };
        // Exactly one terminal callback owns the final drop, even if rx disconnected.
        let terminal = (event.kind != EngineEventKind::Tokens)
            .then(|| unsafe { Box::from_raw(data.cast::<CallbackState>()) });
        let tx = terminal
            .as_deref()
            .unwrap_or_else(|| unsafe { &*data.cast::<CallbackState>() });
        let message = catch_unwind(AssertUnwindSafe(|| match event.kind {
            EngineEventKind::Tokens => {
                let ids = unsafe { slice::from_raw_parts(event.token_ids, event.n_tokens) };
                Ok(Event::Tokens(ids.to_vec()))
            }
            EngineEventKind::Done => Ok(Event::Done(event.telemetry)),
            EngineEventKind::Error => {
                let msg = if event.error_message.is_null() {
                    "engine error without a diagnostic".into()
                } else {
                    unsafe { CStr::from_ptr(event.error_message) }.to_string_lossy()
                };
                Err(EngineError::new(event.status, &msg))
            }
        }))
        .unwrap_or_else(|_| {
            Err(EngineError::new(
                EngineStatus::Internal,
                "callback panicked",
            ))
        });
        let _ = tx.send(message);
    }

    pub struct Engine(NonNull<EngineHandle>);
    // The runtime serializes GPU execution and synchronizes all shared request state.
    // Drop requires exclusive ownership and joins the worker before releasing the handle.
    unsafe impl Send for Engine {}
    unsafe impl Sync for Engine {}
    impl Engine {
        pub fn new(config: &str) -> Result<Self, EngineError> {
            validate_config(config)?;
            let config = CString::new(config).expect("validated config");
            unsafe {
                if engine_abi_version() != FLINT_ENGINE_ABI_VERSION {
                    return Err(EngineError::new(
                        EngineStatus::Internal,
                        "engine ABI mismatch",
                    ));
                }
                let mut err = ptr::null_mut();
                let h = engine_create(config.as_ptr(), &mut err);
                NonNull::new(h).map(Self).ok_or_else(|| EngineError {
                    status: EngineStatus::Internal,
                    message: take_string(err).unwrap_or_else(|| "engine creation failed".into()),
                })
            }
        }
        pub fn stats_json(&self) -> Result<String, EngineError> {
            unsafe { take_string(engine_stats_json(self.0.as_ptr())) }
                .ok_or_else(|| EngineError::new(EngineStatus::Internal, "engine stats unavailable"))
        }
        pub fn submit(&self, req: &Request<'_>) -> Result<(u64, Vec<Event>), EngineError> {
            let params = RequestParams {
                max_tokens: req.max_tokens,
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
                seed: req.seed,
                stop_token_ids: req.stop_token_ids.as_ptr(),
                n_stop_token_ids: req.stop_token_ids.len(),
                ignore_eos: req.ignore_eos.into(),
            };
            let (tx, rx) = mpsc::channel::<Result<Event, EngineError>>();
            let state = Box::into_raw(Box::new(tx));
            let id = unsafe {
                engine_submit(
                    self.0.as_ptr(),
                    &params,
                    req.token_ids.as_ptr(),
                    req.token_ids.len(),
                    ptr::null(),
                    0,
                    collect,
                    state.cast(),
                )
            };
            if id == 0 {
                unsafe { drop(Box::from_raw(state)) };
                return Err(EngineError::new(
                    EngineStatus::Internal,
                    "engine submission failed",
                ));
            }
            let mut events = Vec::new();
            for message in rx {
                let event = message?;
                let done = matches!(event, Event::Done(_));
                events.push(event);
                if done {
                    return Ok((id, events));
                }
            }
            Err(EngineError::new(
                EngineStatus::Internal,
                "missing terminal event",
            ))
        }
        pub fn cancel(&self, id: u64) {
            unsafe { engine_cancel(self.0.as_ptr(), id) }
        }
    }
    impl Drop for Engine {
        fn drop(&mut self) {
            unsafe { engine_destroy(self.0.as_ptr()) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn config_and_empty_prompt_errors() {
        for config in ["", "{\0}"] {
            assert_eq!(
                Engine::new(config).err().unwrap().status,
                EngineStatus::InvalidInput
            );
        }
        let engine = Engine::new("{}").unwrap();
        assert_eq!(
            engine.submit(&Request::default()).unwrap_err().status,
            EngineStatus::InvalidInput
        );
    }

    #[test]
    fn lifecycle_stream_telemetry_and_cancellation() {
        let engine = Engine::new("{}").unwrap();
        let stats = || serde_json::from_str::<Value>(&engine.stats_json().unwrap()).unwrap();
        assert_eq!(stats()["abi_version"], FLINT_ENGINE_ABI_VERSION);
        #[cfg(not(feature = "link-engine"))]
        assert_eq!(stats()["backend"], "stub");
        let request = Request {
            token_ids: &[3, 2, 1],
            max_tokens: 1,
            ..Default::default()
        };
        let (id, events) = engine.submit(&request).unwrap();
        let [Event::Tokens(ids), Event::Done(t)] = events.as_slice() else {
            panic!("unexpected events: {events:?}")
        };
        assert_eq!(ids, &[42]);
        assert_eq!(
            (t.prefill_tokens, t.decode_tokens, t.decode_rounds),
            (3, 1, 1)
        );
        assert!(t.decode_ms.is_finite() && t.decode_ms >= 0.0);
        assert!(t.prefill_ms.is_finite() && t.prefill_ms >= 0.0);
        engine.cancel(id);
        assert_eq!(stats()["cancelled_requests"], 1);
        let (next, events) = engine
            .submit(&Request {
                max_tokens: 0,
                ..request
            })
            .unwrap();
        assert!(next > id);
        assert!(matches!(&events[..], [Event::Done(t)] if t.decode_tokens == 0));
        assert_eq!(stats()["submitted_requests"], 2);
    }
}
