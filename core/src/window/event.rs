use crate::time::Instant;

use std::path::PathBuf;

use winit::dpi::PhysicalPosition;

/*pub trait Pixel: Copy + Into<f64> {
    fn from_f64(f: f64) -> Self;
    fn cast<P: Pixel>(self) -> P {
        P::from_f64(self.into())
    }
}

impl Pixel for u8 {
    fn from_f64(f: f64) -> Self {
        f.round() as u8
    }
}
impl Pixel for u16 {
    fn from_f64(f: f64) -> Self {
        f.round() as u16
    }
}
impl Pixel for u32 {
    fn from_f64(f: f64) -> Self {
        f.round() as u32
    }
}
impl Pixel for i8 {
    fn from_f64(f: f64) -> Self {
        f.round() as i8
    }
}
impl Pixel for i16 {
    fn from_f64(f: f64) -> Self {
        f.round() as i16
    }
}
impl Pixel for i32 {
    fn from_f64(f: f64) -> Self {
        f.round() as i32
    }
}
impl Pixel for f32 {
    fn from_f64(f: f64) -> Self {
        f as f32
    }
}
impl Pixel for f64 {
    fn from_f64(f: f64) -> Self {
        f
    }
}

/// Checks that the scale factor is a normal positive `f64`.
///
/// All functions that take a scale factor assert that this will return `true`. If you're sourcing scale factors from
/// anywhere other than winit, it's recommended to validate them using this function before passing them to winit;
/// otherwise, you risk panics.
#[inline]
pub fn validate_scale_factor(scale_factor: f64) -> bool {
    scale_factor.is_sign_positive() && scale_factor.is_normal()
}

/// A position represented in logical pixels.
///
/// The position is stored as floats, so please be careful. Casting floats to integers truncates the
/// fractional part, which can cause noticable issues. To help with that, an `Into<(i32, i32)>`
/// implementation is provided which does the rounding for you.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, Hash)]
// #[derive(Debug, Copy, Clone, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LogicalPosition<P> {
    pub x: P,
    pub y: P,
}

impl<P> LogicalPosition<P> {
    #[inline]
    pub const fn new(x: P, y: P) -> Self {
        LogicalPosition { x, y }
    }
}

impl<P: Pixel> LogicalPosition<P> {
    #[inline]
    pub fn from_physical<T: Into<PhysicalPosition<X>>, X: Pixel>(
        physical: T,
        scale_factor: f64,
    ) -> Self {
        physical.into().to_logical(scale_factor)
    }

    #[inline]
    pub fn to_physical<X: Pixel>(&self, scale_factor: f64) -> PhysicalPosition<X> {
        assert!(validate_scale_factor(scale_factor));
        let x = self.x.into() * scale_factor;
        let y = self.y.into() * scale_factor;
        PhysicalPosition::new(x, y).cast()
    }

    #[inline]
    pub fn cast<X: Pixel>(&self) -> LogicalPosition<X> {
        LogicalPosition {
            x: self.x.cast(),
            y: self.y.cast(),
        }
    }
}

impl<P: Pixel, X: Pixel> From<(X, X)> for LogicalPosition<P> {
    fn from((x, y): (X, X)) -> LogicalPosition<P> {
        LogicalPosition::new(x.cast(), y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<LogicalPosition<P>> for (X, X) {
    fn from(p: LogicalPosition<P>) -> (X, X) {
        (p.x.cast(), p.y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<[X; 2]> for LogicalPosition<P> {
    fn from([x, y]: [X; 2]) -> LogicalPosition<P> {
        LogicalPosition::new(x.cast(), y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<LogicalPosition<P>> for [X; 2] {
    fn from(p: LogicalPosition<P>) -> [X; 2] {
        [p.x.cast(), p.y.cast()]
    }
}

#[cfg(feature = "mint")]
impl<P: Pixel> From<mint::Point2<P>> for LogicalPosition<P> {
    fn from(p: mint::Point2<P>) -> Self {
        Self::new(p.x, p.y)
    }
}

#[cfg(feature = "mint")]
impl<P: Pixel> From<LogicalPosition<P>> for mint::Point2<P> {
    fn from(p: LogicalPosition<P>) -> Self {
        mint::Point2 { x: p.x, y: p.y }
    }
}

/// A position represented in physical pixels.
// #[derive(Debug, Copy, Clone, Eq, PartialEq, Default, Hash)]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PhysicalPosition<P> {
    pub x: P,
    pub y: P,
}

impl<P> PhysicalPosition<P> {
    #[inline]
    pub const fn new(x: P, y: P) -> Self {
        PhysicalPosition { x, y }
    }
}

impl<P: Pixel> PhysicalPosition<P> {
    #[inline]
    pub fn from_logical<T: Into<LogicalPosition<X>>, X: Pixel>(
        logical: T,
        scale_factor: f64,
    ) -> Self {
        logical.into().to_physical(scale_factor)
    }

    #[inline]
    pub fn to_logical<X: Pixel>(&self, scale_factor: f64) -> LogicalPosition<X> {
        assert!(validate_scale_factor(scale_factor));
        let x = self.x.into() / scale_factor;
        let y = self.y.into() / scale_factor;
        LogicalPosition::new(x, y).cast()
    }

    #[inline]
    pub fn cast<X: Pixel>(&self) -> PhysicalPosition<X> {
        PhysicalPosition {
            x: self.x.cast(),
            y: self.y.cast(),
        }
    }
}

impl<P: Pixel, X: Pixel> From<(X, X)> for PhysicalPosition<P> {
    fn from((x, y): (X, X)) -> PhysicalPosition<P> {
        PhysicalPosition::new(x.cast(), y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<PhysicalPosition<P>> for (X, X) {
    fn from(p: PhysicalPosition<P>) -> (X, X) {
        (p.x.cast(), p.y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<[X; 2]> for PhysicalPosition<P> {
    fn from([x, y]: [X; 2]) -> PhysicalPosition<P> {
        PhysicalPosition::new(x.cast(), y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<PhysicalPosition<P>> for [X; 2] {
    fn from(p: PhysicalPosition<P>) -> [X; 2] {
        [p.x.cast(), p.y.cast()]
    }
}

#[cfg(feature = "mint")]
impl<P: Pixel> From<mint::Point2<P>> for PhysicalPosition<P> {
    fn from(p: mint::Point2<P>) -> Self {
        Self::new(p.x, p.y)
    }
}

#[cfg(feature = "mint")]
impl<P: Pixel> From<PhysicalPosition<P>> for mint::Point2<P> {
    fn from(p: PhysicalPosition<P>) -> Self {
        mint::Point2 { x: p.x, y: p.y }
    }
}

/// A size represented in logical pixels.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LogicalSize<P> {
    pub width: P,
    pub height: P,
}

impl<P> LogicalSize<P> {
    #[inline]
    pub const fn new(width: P, height: P) -> Self {
        LogicalSize { width, height }
    }
}

impl<P: Pixel> LogicalSize<P> {
    #[inline]
    pub fn from_physical<T: Into<PhysicalSize<X>>, X: Pixel>(
        physical: T,
        scale_factor: f64,
    ) -> Self {
        physical.into().to_logical(scale_factor)
    }

    #[inline]
    pub fn to_physical<X: Pixel>(&self, scale_factor: f64) -> PhysicalSize<X> {
        assert!(validate_scale_factor(scale_factor));
        let width = self.width.into() * scale_factor;
        let height = self.height.into() * scale_factor;
        PhysicalSize::new(width, height).cast()
    }

    #[inline]
    pub fn cast<X: Pixel>(&self) -> LogicalSize<X> {
        LogicalSize {
            width: self.width.cast(),
            height: self.height.cast(),
        }
    }
}

impl<P: Pixel, X: Pixel> From<(X, X)> for LogicalSize<P> {
    fn from((x, y): (X, X)) -> LogicalSize<P> {
        LogicalSize::new(x.cast(), y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<LogicalSize<P>> for (X, X) {
    fn from(s: LogicalSize<P>) -> (X, X) {
        (s.width.cast(), s.height.cast())
    }
}

impl<P: Pixel, X: Pixel> From<[X; 2]> for LogicalSize<P> {
    fn from([x, y]: [X; 2]) -> LogicalSize<P> {
        LogicalSize::new(x.cast(), y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<LogicalSize<P>> for [X; 2] {
    fn from(s: LogicalSize<P>) -> [X; 2] {
        [s.width.cast(), s.height.cast()]
    }
}

#[cfg(feature = "mint")]
impl<P: Pixel> From<mint::Vector2<P>> for LogicalSize<P> {
    fn from(v: mint::Vector2<P>) -> Self {
        Self::new(v.x, v.y)
    }
}

#[cfg(feature = "mint")]
impl<P: Pixel> From<LogicalSize<P>> for mint::Vector2<P> {
    fn from(s: LogicalSize<P>) -> Self {
        mint::Vector2 {
            x: s.width,
            y: s.height,
        }
    }
}

/// A size represented in physical pixels.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PhysicalSize<P> {
    pub width: P,
    pub height: P,
}

impl<P> PhysicalSize<P> {
    #[inline]
    pub const fn new(width: P, height: P) -> Self {
        PhysicalSize { width, height }
    }
}

impl<P: Pixel> PhysicalSize<P> {
    #[inline]
    pub fn from_logical<T: Into<LogicalSize<X>>, X: Pixel>(logical: T, scale_factor: f64) -> Self {
        logical.into().to_physical(scale_factor)
    }

    #[inline]
    pub fn to_logical<X: Pixel>(&self, scale_factor: f64) -> LogicalSize<X> {
        assert!(validate_scale_factor(scale_factor));
        let width = self.width.into() / scale_factor;
        let height = self.height.into() / scale_factor;
        LogicalSize::new(width, height).cast()
    }

    #[inline]
    pub fn cast<X: Pixel>(&self) -> PhysicalSize<X> {
        PhysicalSize {
            width: self.width.cast(),
            height: self.height.cast(),
        }
    }
}

impl<P: Pixel, X: Pixel> From<(X, X)> for PhysicalSize<P> {
    fn from((x, y): (X, X)) -> PhysicalSize<P> {
        PhysicalSize::new(x.cast(), y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<PhysicalSize<P>> for (X, X) {
    fn from(s: PhysicalSize<P>) -> (X, X) {
        (s.width.cast(), s.height.cast())
    }
}

impl<P: Pixel, X: Pixel> From<[X; 2]> for PhysicalSize<P> {
    fn from([x, y]: [X; 2]) -> PhysicalSize<P> {
        PhysicalSize::new(x.cast(), y.cast())
    }
}

impl<P: Pixel, X: Pixel> From<PhysicalSize<P>> for [X; 2] {
    fn from(s: PhysicalSize<P>) -> [X; 2] {
        [s.width.cast(), s.height.cast()]
    }
}

#[cfg(feature = "mint")]
impl<P: Pixel> From<mint::Vector2<P>> for PhysicalSize<P> {
    fn from(v: mint::Vector2<P>) -> Self {
        Self::new(v.x, v.y)
    }
}

#[cfg(feature = "mint")]
impl<P: Pixel> From<PhysicalSize<P>> for mint::Vector2<P> {
    fn from(s: PhysicalSize<P>) -> Self {
        mint::Vector2 {
            x: s.width,
            y: s.height,
        }
    }
}

/// A size that's either physical or logical.
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Size {
    Physical(PhysicalSize<u32>),
    Logical(LogicalSize<f64>),
}

impl Size {
    pub fn new<S: Into<Size>>(size: S) -> Size {
        size.into()
    }

    pub fn to_logical<P: Pixel>(&self, scale_factor: f64) -> LogicalSize<P> {
        match *self {
            Size::Physical(size) => size.to_logical(scale_factor),
            Size::Logical(size) => size.cast(),
        }
    }

    pub fn to_physical<P: Pixel>(&self, scale_factor: f64) -> PhysicalSize<P> {
        match *self {
            Size::Physical(size) => size.cast(),
            Size::Logical(size) => size.to_physical(scale_factor),
        }
    }

    pub fn clamp<S: Into<Size>>(input: S, min: S, max: S, scale_factor: f64) -> Size {
        let (input, min, max) = (
            input.into().to_physical::<f64>(scale_factor),
            min.into().to_physical::<f64>(scale_factor),
            max.into().to_physical::<f64>(scale_factor),
        );

        let width = input.width.clamp(min.width, max.width);
        let height = input.height.clamp(min.height, max.height);

        PhysicalSize::new(width, height).into()
    }
}

impl<P: Pixel> From<PhysicalSize<P>> for Size {
    #[inline]
    fn from(size: PhysicalSize<P>) -> Size {
        Size::Physical(size.cast())
    }
}

impl<P: Pixel> From<LogicalSize<P>> for Size {
    #[inline]
    fn from(size: LogicalSize<P>) -> Size {
        Size::Logical(size.cast())
    }
}

/// A position that's either physical or logical.
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Position {
    Physical(PhysicalPosition<i32>),
    Logical(LogicalPosition<f64>),
}

impl Position {
    pub fn new<S: Into<Position>>(position: S) -> Position {
        position.into()
    }

    pub fn to_logical<P: Pixel>(&self, scale_factor: f64) -> LogicalPosition<P> {
        match *self {
            Position::Physical(position) => position.to_logical(scale_factor),
            Position::Logical(position) => position.cast(),
        }
    }

    pub fn to_physical<P: Pixel>(&self, scale_factor: f64) -> PhysicalPosition<P> {
        match *self {
            Position::Physical(position) => position.cast(),
            Position::Logical(position) => position.to_physical(scale_factor),
        }
    }
}

impl<P: Pixel> From<PhysicalPosition<P>> for Position {
    #[inline]
    fn from(position: PhysicalPosition<P>) -> Position {
        Position::Physical(position.cast())
    }
}

impl<P: Pixel> From<LogicalPosition<P>> for Position {
    #[inline]
    fn from(position: LogicalPosition<P>) -> Position {
        Position::Logical(position.cast())
    }
}

impl PartialEq for PhysicalPosition<f64> {
    fn eq(&self, other: &Self) -> bool {
        self.x.eq(&other.x) && self.y.eq(&other.y)
    }
}

impl Eq for PhysicalPosition<f64> {}*/



/// A window-related event.
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Event {
    /// A window was moved.
    Moved {
        /// The new logical x location of the window
        x: i32,
        /// The new logical y location of the window
        y: i32,
    },

    /// A window was resized.
    Resized {
        /// The new logical width of the window
        width: u32,
        /// The new logical height of the window
        height: u32,
    },

    /// A window redraw was requested.
    ///
    /// The [`Instant`] contains the current time.
    RedrawRequested(Instant),

    /// The user has requested for the window to close.
    ///
    /// Usually, you will want to terminate the execution whenever this event
    /// occurs.
    CloseRequested,

    /// A window was focused.
    Focused,

    /// A window was unfocused.
    Unfocused,

    /// A file is being hovered over the window.
    ///
    /// When the user hovers multiple files at once, this event will be emitted
    /// for each file separately.
    FileHovered(PhysicalPosition<u64>), // should be f64 but a hack

    /// A file has beend dropped into the window.
    ///
    /// When the user drops multiple files at once, this event will be emitted
    /// for each file separately.
    FileDropped(Vec<PathBuf>, PhysicalPosition<u64>),

    /// A file was hovered, but has exited the window.
    ///
    /// There will be a single `FilesHoveredLeft` event triggered even if
    /// multiple files were hovered.
    FilesHoveredLeft,
}
