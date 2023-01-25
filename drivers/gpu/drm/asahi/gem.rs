// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(dead_code)]

//! Asahi GEM object implementation

use kernel::{
    bindings, c_str, drm,
    drm::{device, drv, gem, gem::shmem},
    error::{to_result, Result},
    io_mem::IoMem,
    module_platform_driver, of, platform,
    prelude::*,
    soc::apple::rtkit,
    sync::smutex::Mutex,
    sync::{Arc, ArcBorrow},
};

use kernel::drm::gem::BaseObject;

use crate::debug::*;
use crate::driver::AsahiDevice;
use crate::file::DrmFile;

const DEBUG_CLASS: DebugFlags = DebugFlags::Gem;

pub(crate) struct DriverObject {
    kernel: bool,
    flags: u32,
    mappings: Mutex<Vec<(u64, u64, crate::mmu::Mapping)>>,
}

pub(crate) type Object = shmem::Object<DriverObject>;
pub(crate) type SGTable = shmem::SGTable<DriverObject>;

pub(crate) struct ObjectRef {
    pub(crate) gem: gem::ObjectRef<shmem::Object<DriverObject>>,
    vmap: Option<shmem::VMap<DriverObject>>,
}

impl DriverObject {
    fn drop_file_mappings(&self, file_id: u64) {
        let mut mappings = self.mappings.lock();
        for (index, (mapped_fid, _mapped_vmid, _mapping)) in mappings.iter().enumerate() {
            if *mapped_fid == file_id {
                mappings.swap_remove(index);
                return;
            }
        }
    }

    fn drop_vm_mappings(&self, vm_id: u64) {
        let mut mappings = self.mappings.lock();
        for (index, (_mapped_fid, mapped_vmid, _mapping)) in mappings.iter().enumerate() {
            if *mapped_vmid == vm_id {
                mappings.swap_remove(index);
                return;
            }
        }
    }
}

impl ObjectRef {
    pub(crate) fn new(gem: gem::ObjectRef<shmem::Object<DriverObject>>) -> ObjectRef {
        ObjectRef { gem, vmap: None }
    }

    pub(crate) fn vmap(&mut self) -> Result<&mut shmem::VMap<DriverObject>> {
        if self.vmap.is_none() {
            self.vmap = Some(self.gem.vmap()?);
        }
        Ok(self.vmap.as_mut().unwrap())
    }

    pub(crate) fn iova(&self, vm_id: u64) -> Option<usize> {
        let mappings = self.gem.mappings.lock();
        for (_mapped_fid, mapped_vmid, mapping) in mappings.iter() {
            if *mapped_vmid == vm_id {
                return Some(mapping.iova());
            }
        }

        None
    }

    pub(crate) fn size(&self) -> usize {
        self.gem.size()
    }

    pub(crate) fn map_into(&mut self, vm: &crate::mmu::Vm) -> Result<usize> {
        let vm_id = vm.id();
        let mut mappings = self.gem.mappings.lock();
        for (_mapped_fid, mapped_vmid, _mapping) in mappings.iter() {
            if *mapped_vmid == vm_id {
                return Err(EBUSY);
            }
        }

        let sgt = self.gem.sg_table()?;
        let new_mapping = vm.map(self.gem.size(), sgt)?;

        let iova = new_mapping.iova();
        mappings.try_push((vm.file_id(), vm_id, new_mapping))?;
        Ok(iova)
    }

    pub(crate) fn map_into_range(
        &mut self,
        vm: &crate::mmu::Vm,
        start: u64,
        end: u64,
        alignment: u64,
        prot: u32,
        guard: bool,
    ) -> Result<usize> {
        let vm_id = vm.id();
        let mut mappings = self.gem.mappings.lock();
        for (_mapped_fid, mapped_vmid, _mapping) in mappings.iter() {
            if *mapped_vmid == vm_id {
                return Err(EBUSY);
            }
        }

        let sgt = self.gem.sg_table()?;
        let new_mapping =
            vm.map_in_range(self.gem.size(), sgt, alignment, start, end, prot, guard)?;

        let iova = new_mapping.iova();
        mappings.try_push((vm.file_id(), vm_id, new_mapping))?;
        Ok(iova)
    }

    pub(crate) fn map_at(
        &mut self,
        vm: &crate::mmu::Vm,
        addr: u64,
        prot: u32,
        guard: bool,
    ) -> Result {
        let vm_id = vm.id();
        let mut mappings = self.gem.mappings.lock();
        for (_mapped_fid, mapped_vmid, _mapping) in mappings.iter() {
            if *mapped_vmid == vm_id {
                return Err(EBUSY);
            }
        }

        let sgt = self.gem.sg_table()?;
        let new_mapping = vm.map_at(addr, self.gem.size(), sgt, prot, guard)?;

        let iova = new_mapping.iova();
        assert!(iova == addr as usize);
        mappings.try_push((vm.file_id(), vm_id, new_mapping))?;
        Ok(())
    }

    pub(crate) fn drop_file_mappings(&mut self, file_id: u64) {
        self.gem.drop_file_mappings(file_id);
    }

    pub(crate) fn drop_vm_mappings(&mut self, vm_id: u64) {
        self.gem.drop_vm_mappings(vm_id);
    }
}

pub(crate) fn new_kernel_object(dev: &AsahiDevice, size: usize) -> Result<ObjectRef> {
    let mut gem = shmem::Object::<DriverObject>::new(dev, size)?;
    gem.kernel = true;
    gem.flags = 0;

    Ok(ObjectRef::new(gem.into_ref()))
}

pub(crate) fn new_object(dev: &AsahiDevice, size: usize, flags: u32) -> Result<ObjectRef> {
    let mut gem = shmem::Object::<DriverObject>::new(dev, size)?;
    gem.kernel = false;
    gem.flags = flags;

    gem.set_wc(flags & bindings::ASAHI_GEM_WRITEBACK == 0);

    Ok(ObjectRef::new(gem.into_ref()))
}

pub(crate) fn lookup_handle(file: &DrmFile, handle: u32) -> Result<ObjectRef> {
    Ok(ObjectRef::new(shmem::Object::lookup_handle(file, handle)?))
}

impl gem::BaseDriverObject<Object> for DriverObject {
    fn new(_dev: &AsahiDevice, _size: usize) -> Result<DriverObject> {
        mod_pr_debug!("DriverObject::new\n");
        Ok(DriverObject {
            kernel: false,
            flags: 0,
            mappings: Mutex::new(Vec::new()),
        })
    }

    fn close(obj: &Object, file: &DrmFile) {
        mod_pr_debug!("DriverObject::close\n");
        obj.drop_file_mappings(file.inner().file_id());
    }
}

impl shmem::DriverObject for DriverObject {
    type Driver = crate::driver::AsahiDriver;
}

impl rtkit::Buffer for ObjectRef {
    fn iova(&self) -> Result<usize> {
        self.iova(0).ok_or(EIO)
    }
    fn buf(&mut self) -> Result<&mut [u8]> {
        let vmap = self.vmap.as_mut().ok_or(ENOMEM)?;
        Ok(vmap.as_mut_slice())
    }
}
