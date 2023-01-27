// SPDX-License-Identifier: GPL-2.0-only OR MIT

//! GPU command execution queues
//!
//! The AGX GPU firmware schedules GPU work commands out of work queues, which are ring buffers of
//! pointers to work commands. There can be an arbitrary number of work queues. Work queues have an
//! associated type (vertex, fragment, or compute) and may only contain generic commands or commands
//! specific to that type.
//!
//! This module manages queueing work commands into a work queue and submitting them for execution
//! by the firmware. An active work queue needs an event to signal completion of its work, which is
//! owned by what we call a batch. This event then notifies the work queue when work is completed,
//! and that triggers freeing of all resources associated with that work. An idle work queue gives
//! up its associated event.

use crate::debug::*;
use crate::fw::channels::PipeType;
use crate::fw::event::NotifierList;
use crate::fw::types::*;
use crate::fw::workqueue::*;
use crate::{box_in_place, place};
use crate::{channel, event, fw, gpu, object, regs};
use core::sync::atomic::Ordering;
use kernel::{
    bindings,
    prelude::*,
    sync::{smutex, Arc, CondVar, Guard, Mutex, UniqueArc},
    Opaque,
};

const DEBUG_CLASS: DebugFlags = DebugFlags::WorkQueue;

/// An enum of possible errors that might cause a piece of work to fail execution.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum BatchError {
    /// GPU timeout (command execution took too long).
    Timeout,
    /// GPU MMU fault (invalid access).
    Fault(regs::FaultInfo),
    /// Unknown reason.
    Unknown,
    /// Work failed due to an error caused by other concurrent GPU work.
    Killed,
}

impl From<BatchError> for kernel::error::Error {
    fn from(err: BatchError) -> Self {
        match err {
            BatchError::Timeout => ETIMEDOUT,
            // Not EFAULT because that's for userspace faults
            BatchError::Fault(_) => EIO,
            BatchError::Unknown => ENODATA,
            BatchError::Killed => ECANCELED,
        }
    }
}

/// A batch of commands that has been submitted to a workqueue as one unit.
pub(crate) struct Batch {
    value: event::EventValue,
    commands: usize,
    // TODO: make abstraction
    completion: Opaque<bindings::completion>,
    wptr: u32,
    vm_slot: u32,
    error: smutex::Mutex<Option<BatchError>>,
}

/// SAFETY: The bindings::completion is safe to send/share across threads
unsafe impl Send for Batch {}
unsafe impl Sync for Batch {}

impl Batch {
    /// Wait for the batch to complete execution and return the execution status.
    pub(crate) fn wait(&self) -> core::result::Result<(), BatchError> {
        // TODO: Properly abstract this.
        unsafe { bindings::wait_for_completion(self.completion.get()) };
        self.error.lock().map_or(Ok(()), Err)
    }
}

/// Inner data for managing a single work queue.
#[versions(AGX)]
struct WorkQueueInner {
    event_manager: Arc<event::EventManager>,
    info: GpuObject<QueueInfo::ver>,
    new: bool,
    pipe_type: PipeType,
    size: u32,
    wptr: u32,
    pending: Vec<Box<dyn object::OpaqueGpuObject + Send + Sync>>,
    batches: Vec<Arc<Batch>>,
    last_token: Option<event::Token>,
    event: Option<(event::Event, event::EventValue)>,
    priority: u32,
}

/// An instance of a work queue.
#[versions(AGX)]
pub(crate) struct WorkQueue {
    info_pointer: GpuWeakPointer<QueueInfo::ver>,
    inner: Mutex<WorkQueueInner::ver>,
    cond: CondVar,
}

/// The default work queue size.
const WQ_SIZE: u32 = 0x500;

#[versions(AGX)]
impl WorkQueueInner::ver {
    /// Return the GPU done pointer, representing how many work items have been completed by the
    /// GPU.
    fn doneptr(&self) -> u32 {
        self.info
            .state
            .with(|raw, _inner| raw.gpu_doneptr.load(Ordering::Acquire))
    }
}

/// An in-progress batch of commands to be submitted to a WorkQueue. Further commands can be added
/// before submission.
#[versions(AGX)]
pub(crate) struct BatchBuilder<'a> {
    queue: &'a WorkQueue::ver,
    inner: Guard<'a, Mutex<WorkQueueInner::ver>>,
    commands: usize,
    wptr: u32,
    vm_slot: u32,
}

#[versions(AGX)]
impl WorkQueue::ver {
    /// Create a new WorkQueue of a given type and priority.
    pub(crate) fn new(
        alloc: &mut gpu::KernelAllocators,
        event_manager: Arc<event::EventManager>,
        gpu_context: GpuWeakPointer<GpuContextData>,
        notifier_list: GpuWeakPointer<NotifierList>,
        pipe_type: PipeType,
        id: u64,
        priority: u32,
    ) -> Result<Arc<WorkQueue::ver>> {
        let mut info = box_in_place!(QueueInfo::ver {
            state: alloc.shared.new_default::<RingState>()?,
            ring: alloc.shared.array_empty(WQ_SIZE as usize)?,
            gpu_buf: alloc.private.array_empty(0x2c18)?,
        })?;

        info.state.with_mut(|raw, _inner| {
            raw.rb_size = WQ_SIZE;
        });

        let inner = WorkQueueInner::ver {
            event_manager,
            info: alloc.private.new_boxed(info, |inner, ptr| {
                Ok(place!(
                    ptr,
                    raw::QueueInfo::ver {
                        state: inner.state.gpu_pointer(),
                        ring: inner.ring.gpu_pointer(),
                        notifier_list: notifier_list,
                        gpu_buf: inner.gpu_buf.gpu_pointer(),
                        gpu_rptr1: Default::default(),
                        gpu_rptr2: Default::default(),
                        gpu_rptr3: Default::default(),
                        event_id: AtomicI32::new(-1),
                        priority: *raw::PRIORITY.get(priority as usize).ok_or(EINVAL)?,
                        unk_4c: -1,
                        uuid: id as u32,
                        unk_54: -1,
                        unk_58: Default::default(),
                        busy: Default::default(),
                        __pad: Default::default(),
                        unk_84_state: Default::default(),
                        unk_88: 0,
                        unk_8c: 0,
                        unk_90: 0,
                        unk_94: 0,
                        pending: Default::default(),
                        unk_9c: 0,
                        #[ver(V >= V13_2)]
                        unk_a0_0: 0,
                        gpu_context: gpu_context,
                        unk_a8: Default::default(),
                        #[ver(V >= V13_2)]
                        unk_b0: 0,
                    }
                ))
            })?,
            new: true,
            pipe_type,
            size: WQ_SIZE,
            wptr: 0,
            pending: Vec::new(),
            batches: Vec::new(),
            last_token: None,
            event: None,
            priority,
        };

        let mut queue = Pin::from(UniqueArc::try_new(Self {
            info_pointer: inner.info.weak_pointer(),
            // SAFETY: `condvar_init!` is called below.
            cond: unsafe { CondVar::new() },
            // SAFETY: `mutex_init!` is called below.
            inner: unsafe { Mutex::new(inner) },
        })?);

        // SAFETY: `cond` is pinned when `queue` is.
        let pinned = unsafe { queue.as_mut().map_unchecked_mut(|s| &mut s.cond) };
        match pipe_type {
            PipeType::Vertex => kernel::condvar_init!(pinned, "WorkQueue::cond (Vertex)"),
            PipeType::Fragment => kernel::condvar_init!(pinned, "WorkQueue::cond (Fragment)"),
            PipeType::Compute => kernel::condvar_init!(pinned, "WorkQueue::cond (Compute)"),
        }

        // SAFETY: `inner` is pinned when `queue` is.
        let pinned = unsafe { queue.as_mut().map_unchecked_mut(|s| &mut s.inner) };
        match pipe_type {
            PipeType::Vertex => kernel::mutex_init!(pinned, "WorkQueue::inner (Vertex)"),
            PipeType::Fragment => kernel::mutex_init!(pinned, "WorkQueue::inner (Fragment)"),
            PipeType::Compute => kernel::mutex_init!(pinned, "WorkQueue::inner (Compute)"),
        }

        Ok(queue.into())
    }

    /// Returns the QueueInfo pointer for this workqueue, as a weak pointer.
    pub(crate) fn info_pointer(&self) -> GpuWeakPointer<QueueInfo::ver> {
        self.info_pointer
    }

    /// Start a new batch of work on this queue.
    pub(crate) fn begin_batch(
        this: &Arc<WorkQueue::ver>,
        vm_slot: u32,
    ) -> Result<BatchBuilder::ver<'_>> {
        let mut inner = this.inner.lock();

        if inner.event.is_none() {
            let event = inner.event_manager.get(inner.last_token, this.clone())?;
            let cur = event.current();
            inner.last_token = Some(event.token());
            inner.event = Some((event, cur));
        }

        Ok(BatchBuilder::ver {
            queue: this,
            wptr: inner.wptr,
            inner,
            commands: 0,
            vm_slot,
        })
    }
}

/// Trait used to erase the version-specific type of WorkQueues, to avoid leaking
/// version-specificity into the event module.
pub(crate) trait WorkQueue {
    fn signal(&self) -> bool;
    fn mark_error(&self, value: event::EventValue, error: BatchError);
}

#[versions(AGX)]
impl WorkQueue for WorkQueue::ver {
    /// Signal a workqueue that some work was completed.
    ///
    /// This will check the event stamp value to find out exactly how many commands were processed.
    fn signal(&self) -> bool {
        let mut inner = self.inner.lock();
        let event = inner.event.as_ref();
        let cur_value = match event {
            None => {
                pr_err!("WorkQueue: signal() called but no event?");
                return true;
            }
            Some(event) => event.0.current(),
        };

        mod_pr_debug!(
            "WorkQueue({:?}): Signaling event {:?} value {:#x?}",
            inner.pipe_type,
            inner.last_token,
            cur_value
        );

        let mut completed_commands: usize = 0;
        let mut batches: usize = 0;

        for batch in inner.batches.iter() {
            if batch.value <= cur_value {
                mod_pr_debug!(
                    "WorkQueue({:?}): Batch at value {:#x?} complete",
                    inner.pipe_type,
                    batch.value
                );
                completed_commands += batch.commands;
                batches += 1;
            } else {
                break;
            }
        }
        mod_pr_debug!(
            "WorkQueue({:?}): Completed {} batches",
            inner.pipe_type,
            batches
        );

        let mut completed = Vec::new();
        for i in inner.batches.drain(..batches) {
            if completed.try_push(i).is_err() {
                pr_err!("Failed to signal completions");
                break;
            }
        }
        if let Some(i) = completed.last() {
            inner
                .info
                .state
                .with(|raw, _inner| raw.cpu_freeptr.store(i.wptr, Ordering::Release));
        }

        inner.pending.drain(..completed_commands);
        self.cond.notify_all();
        let empty = inner.batches.is_empty();
        if empty {
            inner.event = None;
        }
        core::mem::drop(inner);

        for batch in completed {
            // TODO: Properly abstract this.
            unsafe { bindings::complete_all(batch.completion.get()) };
        }
        empty
    }

    /// Mark this queue's work up to a certain stamp value as having failed.
    fn mark_error(&self, value: event::EventValue, error: BatchError) {
        // If anything is marked completed, we can consider it successful
        // at this point, even if we didn't get the signal event yet.
        self.signal();

        let inner = self.inner.lock();

        if inner.event.is_none() {
            pr_err!("WorkQueue: signal_fault() called but no event?");
            return;
        }

        mod_pr_debug!(
            "WorkQueue({:?}): Signaling fault for event {:?} at value {:#x?}",
            inner.pipe_type,
            inner.last_token,
            value
        );

        for batch in inner.batches.iter() {
            if batch.value <= value {
                mod_pr_debug!(
                    "WorkQueue({:?}): Batch at value {:#x?} failed ({} commands)",
                    inner.pipe_type,
                    batch.value,
                    batch.commands,
                );
                *(batch.error.lock()) = Some(match error {
                    BatchError::Fault(info) if info.vm_slot != batch.vm_slot => BatchError::Killed,
                    err => err,
                });
            } else {
                break;
            }
        }
    }
}

#[versions(AGX)]
impl<'a> BatchBuilder::ver<'a> {
    /// Add a command to a work batch.
    pub(crate) fn add<T: Command>(&mut self, command: Box<GpuObject<T>>) -> Result {
        let inner = &mut self.inner;

        let next_wptr = (self.wptr + 1) % inner.size;
        if inner.doneptr() == next_wptr {
            pr_err!("Work queue ring buffer is full! Waiting...");
            while inner.doneptr() == next_wptr {
                if self.queue.cond.wait(inner) {
                    return Err(ERESTARTSYS);
                }
            }
        }
        inner.pending.try_reserve(1)?;

        inner.info.ring[self.wptr as usize] = command.gpu_va().get();

        self.wptr = next_wptr;

        // Cannot fail, since we did a try_reserve(1) above
        inner
            .pending
            .try_push(command)
            .expect("try_push() failed after try_reserve(1)");
        self.commands += 1;
        Ok(())
    }

    /// Commit the pending commands and submit them to the GPU, returning a Batch object. This
    /// builder can then be reused to submit more commands.
    ///
    /// Note that the GPU must still be notified separately to actually begin work execution on any
    /// given queue by using GpuManager::submit_batch().
    pub(crate) fn commit(&mut self) -> Result<Arc<Batch>> {
        let inner = &mut self.inner;
        inner.batches.try_reserve(1)?;

        let event = inner.event.as_mut().expect("BatchBuilder lost its event");

        if self.commands == 0 {
            return Err(EINVAL);
        }

        event.1.increment();
        let event_value = event.1;

        inner
            .info
            .state
            .with(|raw, _inner| raw.cpu_wptr.store(self.wptr, Ordering::Release));

        inner.wptr = self.wptr;
        let batch = Arc::try_new(Batch {
            value: event_value,
            commands: self.commands,
            completion: Opaque::uninit(),
            wptr: self.wptr,
            error: smutex::Mutex::new(None),
            vm_slot: self.vm_slot,
        })?;
        unsafe { bindings::init_completion(batch.completion.get()) };
        inner.batches.try_push(batch.clone())?;
        self.commands = 0;
        Ok(batch)
    }

    /// Submit a work execution request for the newest committed batch to a PipeChannel.
    ///
    /// All pending work must have been committed before calling this.
    pub(crate) fn submit(mut self, channel: &mut channel::PipeChannel::ver) -> Result {
        if self.commands != 0 {
            return Err(EINVAL);
        }

        let inner = &mut self.inner;
        let event = inner.event.as_ref().expect("BatchBuilder lost its event");
        let msg = fw::channels::RunWorkQueueMsg::ver {
            pipe_type: inner.pipe_type,
            work_queue: Some(inner.info.weak_pointer()),
            wptr: inner.wptr,
            event_slot: event.0.slot(),
            is_new: inner.new,
            __pad: Default::default(),
        };
        channel.send(&msg);
        inner.new = false;
        Ok(())
    }

    /// Return the Event associated with this in-progress batch.
    pub(crate) fn event(&self) -> &event::Event {
        let event = self
            .inner
            .event
            .as_ref()
            .expect("BatchBuilder lost its event");
        &(event.0)
    }

    /// Returns the current base event value associated with this in-progress batch.
    ///
    /// New work should increment this and use it as the completion value.
    pub(crate) fn event_value(&self) -> event::EventValue {
        let event = self
            .inner
            .event
            .as_ref()
            .expect("BatchBuilder lost its event");
        event.1
    }

    /// Returns the pipe type of this queue.
    pub(crate) fn pipe_type(&self) -> PipeType {
        self.inner.pipe_type
    }

    /// Returns the priority of this queue.
    pub(crate) fn priority(&self) -> u32 {
        self.inner.priority
    }
}

#[versions(AGX)]
impl<'a> Drop for BatchBuilder::ver<'a> {
    fn drop(&mut self) {
        if self.commands != 0 {
            pr_warn!("BatchBuilder: rolling back {} commands!", self.commands);

            let inner = &mut self.inner;
            let new_len = inner.pending.len() - self.commands;
            inner.pending.truncate(new_len);
        }
    }
}
