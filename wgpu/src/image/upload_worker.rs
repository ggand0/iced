use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc::{channel, Sender, Receiver};

pub struct UploadWorker {
    sender: Sender<UploadJob>,
    _worker_thread: thread::JoinHandle<()>,
}

struct UploadJob {
    data: Vec<u8>,
    width: u32,
    height: u32,
    handle_id: u64,
    callback: Arc<Mutex<dyn FnMut(u64, Vec<u8>) + Send + 'static>>,
}

impl UploadWorker {
    pub fn new() -> Self {
        let (sender, receiver) = channel();
        
        let worker_thread = thread::spawn(move || {
            Self::worker_loop(receiver);
        });
        
        Self {
            sender,
            _worker_thread: worker_thread,
        }
    }
    
    fn worker_loop(receiver: Receiver<UploadJob>) {
        while let Ok(job) = receiver.recv() {
            // Process the image data on a background thread
            // This could include resizing, format conversion, etc.
            
            // When finished, notify through the callback
            if let Ok(mut callback) = job.callback.lock() {
                callback(job.handle_id, job.data);
            }
        }
    }
    
    pub fn queue_upload(
        &self,
        data: Vec<u8>,
        width: u32,
        height: u32,
        handle_id: u64,
        callback: Arc<Mutex<dyn FnMut(u64, Vec<u8>) + Send + 'static>>,
    ) {
        let job = UploadJob {
            data,
            width,
            height,
            handle_id,
            callback,
        };
        
        let _ = self.sender.send(job);
    }
}