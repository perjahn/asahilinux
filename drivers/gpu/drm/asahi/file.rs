// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(clippy::unusual_byte_groupings)]

//! Asahi File state

use crate::debug::*;
use crate::driver::AsahiDevice;
use crate::fw::types::*;
use crate::{alloc, buffer, driver, gem, gpu, mmu, render};
use kernel::drm::gem::BaseObject;
use kernel::prelude::*;
use kernel::sync::{smutex::Mutex, Arc};
use kernel::{bindings, drm, xarray};

const DEBUG_CLASS: DebugFlags = DebugFlags::File;

struct Vm {
    dummy_obj: gem::ObjectRef,
    ualloc: Arc<Mutex<alloc::DefaultAllocator>>,
    ualloc_priv: Arc<Mutex<alloc::DefaultAllocator>>,
    vm: mmu::Vm,
}

pub(crate) trait Queue: Send + Sync {
    fn submit(&self, cmd: &bindings::drm_asahi_submit, id: u64) -> Result;
}

pub(crate) struct File {
    id: u64,
    vms: xarray::XArray<Box<Vm>>,
    queues: xarray::XArray<Arc<Box<dyn Queue>>>,
}

pub(crate) type DrmFile = drm::file::File<File>;

const VM_SHADER_START: u64 = 0x11_00000000;
const VM_SHADER_END: u64 = 0x11_ffffffff;
const VM_USER_START: u64 = 0x20_00000000;
const VM_USER_END: u64 = 0x5f_ffffffff;

const VM_DRV_GPU_START: u64 = 0x60_00000000;
const VM_DRV_GPU_END: u64 = 0x60_ffffffff;
const VM_DRV_GPUFW_START: u64 = 0x61_00000000;
const VM_DRV_GPUFW_END: u64 = 0x61_ffffffff;
const VM_UNK_PAGE: u64 = 0x6f_ffff8000;

impl drm::file::DriverFile for File {
    type Driver = driver::AsahiDriver;

    fn open(device: &AsahiDevice) -> Result<Box<Self>> {
        debug::update_debug_flags();

        let gpu = &device.data().gpu;
        let id = gpu.ids().file.next();

        mod_dev_dbg!(device, "[File {}]: DRM device opened", id);
        Ok(Box::try_new(Self {
            id,
            vms: xarray::XArray::new(xarray::flags::ALLOC1)?,
            queues: xarray::XArray::new(xarray::flags::ALLOC1)?,
        })?)
    }
}

macro_rules! param {
    ($name:ident) => {
        kernel::macros::concat_idents!(bindings::drm_asahi_param_DRM_ASAHI_PARAM_, $name)
    };
}

impl File {
    pub(crate) fn get_param(
        device: &AsahiDevice,
        data: &mut bindings::drm_asahi_get_param,
        file: &DrmFile,
    ) -> Result<u32> {
        mod_dev_dbg!(device, "[File {}]: IOCTL: get_param", file.inner().id);

        let gpu = &device.data().gpu;

        let value: u64 = match data.param {
            param!(UNSTABLE_UABI_VERSION) => bindings::DRM_ASAHI_UNSTABLE_UABI_VERSION as u64,
            param!(GPU_GENERATION) => gpu.get_dyncfg().id.gpu_gen as u32 as u64,
            param!(GPU_VARIANT) => gpu.get_dyncfg().id.gpu_variant as u32 as u64,
            param!(GPU_REVISION) => gpu.get_dyncfg().id.gpu_rev as u32 as u64,
            param!(CHIP_ID) => gpu.get_cfg().chip_id as u64,
            param!(FEAT_COMPAT) => gpu.get_cfg().gpu_feat_compat,
            param!(FEAT_INCOMPAT) => gpu.get_cfg().gpu_feat_incompat,
            param!(VM_PAGE_SIZE) => mmu::UAT_PGSZ as u64,
            param!(VM_USER_START) => VM_USER_START,
            param!(VM_USER_END) => VM_USER_END,
            param!(VM_SHADER_START) => VM_SHADER_START,
            param!(VM_SHADER_END) => VM_SHADER_END,
            _ => return Err(EINVAL),
        };

        data.value = value;

        Ok(0)
    }

    pub(crate) fn vm_create(
        device: &AsahiDevice,
        data: &mut bindings::drm_asahi_vm_create,
        file: &DrmFile,
    ) -> Result<u32> {
        let gpu = &device.data().gpu;
        let file_id = file.inner().id;
        let vm = gpu.new_vm(file_id)?;

        let resv = file.inner().vms.reserve()?;
        let id: u32 = resv.index().try_into()?;

        mod_dev_dbg!(device, "[File {} VM {}]: VM Create", file_id, id);
        mod_dev_dbg!(device, "[File {} VM {}]: Creating allocators", file_id, id);
        let ualloc = Arc::try_new(Mutex::new(alloc::DefaultAllocator::new(
            device,
            &vm,
            VM_DRV_GPU_START,
            VM_DRV_GPU_END,
            buffer::PAGE_SIZE,
            mmu::PROT_GPU_SHARED_RW,
            512 * 1024,
            true,
            fmt!("File {} VM {} GPU Shared", file_id, id),
            false,
        )?))?;
        let ualloc_priv = Arc::try_new(Mutex::new(alloc::DefaultAllocator::new(
            device,
            &vm,
            VM_DRV_GPUFW_START,
            VM_DRV_GPUFW_END,
            buffer::PAGE_SIZE,
            mmu::PROT_GPU_FW_PRIV_RW,
            64 * 1024,
            true,
            fmt!("File {} VM {} GPU FW Private", file_id, id),
            false,
        )?))?;

        mod_dev_dbg!(
            device,
            "[File {} VM {}]: Creating dummy object",
            file_id,
            id
        );
        let mut dummy_obj = gem::new_kernel_object(device, 0x4000)?;
        dummy_obj.vmap()?.as_mut_slice().fill(0);
        dummy_obj.map_at(&vm, VM_UNK_PAGE, mmu::PROT_GPU_SHARED_RW, true)?;

        mod_dev_dbg!(device, "[File {} VM {}]: VM created", file_id, id);
        resv.store(Box::try_new(Vm {
            dummy_obj,
            ualloc,
            ualloc_priv,
            vm,
        })?)?;

        data.vm_id = id;

        Ok(0)
    }

    pub(crate) fn vm_destroy(
        _device: &AsahiDevice,
        data: &mut bindings::drm_asahi_vm_destroy,
        file: &DrmFile,
    ) -> Result<u32> {
        if file.inner().vms.remove(data.vm_id as usize).is_none() {
            Err(ENOENT)
        } else {
            Ok(0)
        }
    }

    pub(crate) fn gem_create(
        device: &AsahiDevice,
        data: &mut bindings::drm_asahi_gem_create,
        file: &DrmFile,
    ) -> Result<u32> {
        mod_dev_dbg!(
            device,
            "[File {}]: IOCTL: gem_create size={:#x?}",
            file.inner().id,
            data.size
        );

        if (data.flags & !bindings::ASAHI_GEM_WRITEBACK) != 0 {
            return Err(EINVAL);
        }

        let bo = gem::new_object(device, data.size.try_into()?, data.flags)?;

        let handle = bo.gem.create_handle(file)?;
        data.handle = handle;

        mod_dev_dbg!(
            device,
            "[File {}]: IOCTL: gem_create size={:#x} handle={:#x?}",
            file.inner().id,
            data.size,
            data.handle
        );

        Ok(0)
    }

    pub(crate) fn gem_mmap_offset(
        device: &AsahiDevice,
        data: &mut bindings::drm_asahi_gem_mmap_offset,
        file: &DrmFile,
    ) -> Result<u32> {
        mod_dev_dbg!(
            device,
            "[File {}]: IOCTL: gem_mmap_offset handle={:#x?}",
            file.inner().id,
            data.handle
        );

        if data.flags != 0 {
            return Err(EINVAL);
        }

        let bo = gem::lookup_handle(file, data.handle)?;
        data.offset = bo.gem.create_mmap_offset()?;
        Ok(0)
    }

    pub(crate) fn gem_bind(
        device: &AsahiDevice,
        data: &mut bindings::drm_asahi_gem_bind,
        file: &DrmFile,
    ) -> Result<u32> {
        mod_dev_dbg!(
            device,
            "[File {} VM {}]: IOCTL: gem_bind handle={:#x?} flags={:#x?} {:#x?}:{:#x?} -> {:#x?}",
            file.inner().id,
            data.vm_id,
            data.handle,
            data.flags,
            data.offset,
            data.range,
            data.addr
        );

        if data.offset != 0 {
            return Err(EINVAL); // Not supported yet
        }

        if (data.addr | data.range) as usize & mmu::UAT_PGMSK != 0 {
            return Err(EINVAL); // Must be page aligned
        }

        if (data.flags & !(bindings::ASAHI_BIND_READ | bindings::ASAHI_BIND_WRITE)) != 0 {
            return Err(EINVAL);
        }

        let mut bo = gem::lookup_handle(file, data.handle)?;

        if data.range != bo.size().try_into()? {
            return Err(EINVAL); // Not supported yet
        }

        let start = data.addr;
        let end = data.addr + data.range - 1;

        if (VM_SHADER_START..=VM_SHADER_END).contains(&start) {
            if !(VM_SHADER_START..=VM_SHADER_END).contains(&end) {
                return Err(EINVAL); // Invalid map range
            }
        } else if (VM_USER_START..=VM_USER_END).contains(&start) {
            if !(VM_USER_START..=VM_USER_END).contains(&end) {
                return Err(EINVAL); // Invalid map range
            }
        } else {
            return Err(EINVAL); // Invalid map range
        }

        // Just in case
        if end >= VM_DRV_GPU_START {
            return Err(EINVAL);
        }

        let prot = if data.flags & bindings::ASAHI_BIND_READ != 0 {
            if data.flags & bindings::ASAHI_BIND_WRITE != 0 {
                mmu::PROT_GPU_SHARED_RW
            } else {
                mmu::PROT_GPU_SHARED_RO
            }
        } else if data.flags & bindings::ASAHI_BIND_WRITE != 0 {
            mmu::PROT_GPU_SHARED_WO
        } else {
            return Err(EINVAL); // Must specify one of ASAHI_BIND_{READ,WRITE}
        };

        // Clone it immediately so we aren't holding the XArray lock
        let vm = file
            .inner()
            .vms
            .get(data.vm_id.try_into()?)
            .ok_or(ENOENT)?
            .vm
            .clone();

        bo.map_at(&vm, start, prot, true)?;

        Ok(0)
    }

    pub(crate) fn queue_create(
        device: &AsahiDevice,
        data: &mut bindings::drm_asahi_queue_create,
        file: &DrmFile,
    ) -> Result<u32> {
        let file_id = file.inner().id;

        mod_dev_dbg!(
            device,
            "[File {} VM {}]: Creating queue type={:?} prio={:?} flags={:#x?}",
            file_id,
            data.vm_id,
            data.queue_type,
            data.priority,
            data.flags,
        );

        if data.flags != 0 {
            return Err(EINVAL);
        }

        if data.priority > 3 {
            return Err(EINVAL);
        }

        let resv = file.inner().queues.reserve()?;
        let file_vm = file.inner().vms.get(data.vm_id.try_into()?).ok_or(ENOENT)?;
        let vm = file_vm.vm.clone();
        let ualloc = file_vm.ualloc.clone();
        let ualloc_priv = file_vm.ualloc_priv.clone();
        // Drop the vms lock eagerly
        core::mem::drop(file_vm);

        let queue = match data.queue_type {
            bindings::drm_asahi_queue_type_DRM_ASAHI_QUEUE_RENDER => device
                .data()
                .gpu
                .new_render_queue(vm, ualloc, ualloc_priv, data.priority)?,
            bindings::drm_asahi_queue_type_DRM_ASAHI_QUEUE_COMPUTE => device
                .data()
                .gpu
                .new_compute_queue(vm, ualloc, data.priority)?,
            _ => return Err(EINVAL),
        };

        data.queue_id = resv.index().try_into()?;
        resv.store(Arc::try_new(queue)?)?;

        Ok(0)
    }

    pub(crate) fn queue_destroy(
        _device: &AsahiDevice,
        data: &mut bindings::drm_asahi_queue_destroy,
        file: &DrmFile,
    ) -> Result<u32> {
        if file.inner().queues.remove(data.queue_id as usize).is_none() {
            Err(ENOENT)
        } else {
            Ok(0)
        }
    }

    pub(crate) fn submit(
        device: &AsahiDevice,
        data: &mut bindings::drm_asahi_submit,
        file: &DrmFile,
    ) -> Result<u32> {
        debug::update_debug_flags();

        let gpu = &device.data().gpu;
        gpu.update_globals();

        /* Upgrade to Arc<T> to drop the XArray lock early */
        let queue: Arc<Box<dyn Queue>> = file
            .inner()
            .queues
            .get(data.queue_id.try_into()?)
            .ok_or(ENOENT)?
            .borrow()
            .into();

        let id = gpu.ids().submission.next();
        mod_dev_dbg!(
            device,
            "[File {} Queue {}]: IOCTL: submit (submission ID: {})",
            file.inner().id,
            data.queue_id,
            id
        );
        let ret = queue.submit(data, id);
        if let Err(e) = ret {
            dev_info!(
                device,
                "[File {} Queue {}]: IOCTL: submit failed! (submission ID: {} err: {:?})",
                file.inner().id,
                data.queue_id,
                id,
                e
            );
            Err(e)
        } else {
            Ok(0)
        }
    }

    pub(crate) fn file_id(&self) -> u64 {
        self.id
    }
}

impl Drop for File {
    fn drop(&mut self) {
        mod_pr_debug!("[File {}]: Closing...", self.id);
    }
}
