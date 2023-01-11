// SPDX-License-Identifier: GPL-2.0 OR MIT
#![allow(missing_docs)]

//! XArray abstraction
//!
//! C header: [`include/linux/xarray.h`](../../include/linux/xarray.h)

use crate::{bindings, Error, Opaque, PointerWrapper, Result};

use core::{marker::PhantomData, ops::Deref};

pub struct XArray<T: PointerWrapper> {
    xa: Opaque<bindings::xarray>,
    _p: PhantomData<T>,
}

pub struct Guard<'a, T: PointerWrapper>(*mut T, &'a Opaque<bindings::xarray>);

pub struct Reservation<'a, T: PointerWrapper>(&'a XArray<T>, usize, PhantomData<T>);

type Flags = bindings::gfp_t;

pub mod flags {
    pub const LOCK_IRQ: super::Flags = bindings::BINDINGS_XA_FLAGS_LOCK_IRQ;
    pub const LOCK_BH: super::Flags = bindings::BINDINGS_XA_FLAGS_LOCK_BH;
    pub const TRACK_FREE: super::Flags = bindings::BINDINGS_XA_FLAGS_TRACK_FREE;
    pub const ZERO_BUSY: super::Flags = bindings::BINDINGS_XA_FLAGS_ZERO_BUSY;
    pub const ALLOC_WRAPPED: super::Flags = bindings::BINDINGS_XA_FLAGS_ALLOC_WRAPPED;
    pub const ACCOUNT: super::Flags = bindings::BINDINGS_XA_FLAGS_ACCOUNT;
    pub const ALLOC: super::Flags = bindings::BINDINGS_XA_FLAGS_ALLOC;
    pub const ALLOC1: super::Flags = bindings::BINDINGS_XA_FLAGS_ALLOC1;
}

impl<'a, T: PointerWrapper> Guard<'a, T> {
    pub fn borrow<'b>(&'b self) -> T::Borrowed<'b>
    where
        'a: 'b,
    {
        unsafe { T::borrow(self.0 as _) }
    }
}

impl<'a, T: PointerWrapper> Deref for Guard<'a, T>
where
    T::Borrowed<'static>: Deref,
{
    type Target = <T::Borrowed<'static> as Deref>::Target;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(T::borrow(self.0 as _).deref() as *const _) }
    }
}

impl<'a, T: PointerWrapper> Drop for Guard<'a, T> {
    fn drop(&mut self) {
        unsafe { bindings::xa_unlock(self.1.get()) };
    }
}

impl<T: PointerWrapper> XArray<T> {
    pub fn new(flags: Flags) -> Result<XArray<T>> {
        let xa = Opaque::uninit();

        unsafe {
            bindings::xa_init_flags(xa.get(), flags);
        }

        Ok(XArray {
            xa,
            _p: PhantomData,
        })
    }

    pub fn replace(&self, index: usize, value: T) -> Result<Option<T>> {
        let new = value.into_pointer();

        let old = unsafe {
            bindings::xa_store(
                self.xa.get(),
                index.try_into()?,
                new as *mut _,
                bindings::GFP_KERNEL,
            )
        };

        let err = unsafe { bindings::xa_err(old) };
        if err != 0 {
            // Make sure to drop the value we failed to store
            unsafe { T::from_pointer(new) };
            Err(Error::from_kernel_errno(err))
        } else if old.is_null() {
            Ok(None)
        } else {
            Ok(Some(unsafe { T::from_pointer(old) }))
        }
    }

    pub fn set(&self, index: usize, value: T) -> Result {
        self.replace(index, value)?;
        Ok(())
    }

    pub fn get(&self, index: usize) -> Option<Guard<'_, T>> {
        let p = unsafe {
            bindings::xa_lock(self.xa.get());
            bindings::xa_load(self.xa.get(), index.try_into().ok()?)
        };

        if p.is_null() {
            unsafe { bindings::xa_lock(self.xa.get()) };
            None
        } else {
            Some(Guard(p as _, &self.xa))
        }
    }

    pub fn remove(&self, index: usize) -> Option<T> {
        let p = unsafe { bindings::xa_erase(self.xa.get(), index.try_into().ok()?) };
        if p.is_null() {
            None
        } else {
            Some(unsafe { T::from_pointer(p) })
        }
    }

    pub fn alloc_limits(&self, value: Option<T>, min: u32, max: u32) -> Result<usize> {
        let new = value.map_or(core::ptr::null(), |a| a.into_pointer());
        let mut id: u32 = 0;

        let ret = unsafe {
            bindings::xa_alloc(
                self.xa.get(),
                &mut id,
                new as *mut _,
                bindings::xa_limit { min, max },
                bindings::GFP_KERNEL,
            )
        };

        if ret < 0 {
            // Make sure to drop the value we failed to store
            if !new.is_null() {
                unsafe { T::from_pointer(new) };
            }
            Err(Error::from_kernel_errno(ret))
        } else {
            Ok(id as usize)
        }
    }

    pub fn alloc(&self, value: Option<T>) -> Result<usize> {
        self.alloc_limits(value, 0, u32::MAX)
    }

    pub fn reserve_limits(&self, min: u32, max: u32) -> Result<Reservation<'_, T>> {
        Ok(Reservation(
            self,
            self.alloc_limits(None, min, max)?,
            PhantomData,
        ))
    }

    pub fn reserve(&self) -> Result<Reservation<'_, T>> {
        Ok(Reservation(self, self.alloc(None)?, PhantomData))
    }
}

impl<'a, T: PointerWrapper> Reservation<'a, T> {
    pub fn store(self, value: T) -> Result<usize> {
        if self.0.replace(self.1, value)?.is_some() {
            crate::pr_err!("XArray: Reservation stored but the entry already had data!\n");
            // Consider it a success anyway, not much we can do
        }
        let index = self.1;
        core::mem::forget(self);
        Ok(index)
    }

    pub fn index(&self) -> usize {
        self.1 as usize
    }
}

impl<'a, T: PointerWrapper> Drop for Reservation<'a, T> {
    fn drop(&mut self) {
        if self.0.remove(self.1).is_some() {
            crate::pr_err!("XArray: Reservation dropped but the entry was not empty!\n");
        }
    }
}

impl<T: PointerWrapper> Drop for XArray<T> {
    fn drop(&mut self) {
        unsafe {
            let mut index: core::ffi::c_ulong = 0;
            let mut entry = bindings::xa_find(
                self.xa.get(),
                &mut index,
                core::ffi::c_ulong::MAX,
                bindings::BINDINGS_XA_PRESENT,
            );
            while !entry.is_null() {
                T::from_pointer(entry);
                entry = bindings::xa_find_after(
                    self.xa.get(),
                    &mut index,
                    core::ffi::c_ulong::MAX,
                    bindings::BINDINGS_XA_PRESENT,
                );
            }

            bindings::xa_destroy(self.xa.get());
        }
    }
}

unsafe impl<T: Send + PointerWrapper> Send for XArray<T> {}
unsafe impl<T: Sync + PointerWrapper> Sync for XArray<T> {}
