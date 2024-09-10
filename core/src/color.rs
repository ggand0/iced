use palette::rgb::{Srgb, Srgba};

#[derive(Debug, thiserror::Error)]
/// Errors that can occur when constructing a [`Color`].
pub enum ColorError {
    #[error("The specified hex string is invalid. See supported formats.")]
    /// The specified hex string is invalid. See supported formats.
    InvalidHex,
}

/// A color in the `sRGB` color space.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Color {
    /// Red component, 0.0 - 1.0
    pub r: f32,
    /// Green component, 0.0 - 1.0
    pub g: f32,
    /// Blue component, 0.0 - 1.0
    pub b: f32,
    /// Transparency, 0.0 - 1.0
    pub a: f32,
}

impl Color {
    /// The black color.
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };

    /// The white color.
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    /// A color with no opacity.
    pub const TRANSPARENT: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    /// Creates a new [`Color`].
    ///
    /// In debug mode, it will panic if the values are not in the correct
    /// range: 0.0 - 1.0
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
        debug_assert!(
            (0.0..=1.0).contains(&r),
            "Red component must be on [0, 1]"
        );
        debug_assert!(
            (0.0..=1.0).contains(&g),
            "Green component must be on [0, 1]"
        );
        debug_assert!(
            (0.0..=1.0).contains(&b),
            "Blue component must be on [0, 1]"
        );
        debug_assert!(
            (0.0..=1.0).contains(&a),
            "Alpha component must be on [0, 1]"
        );

        Color { r, g, b, a }
    }

    /// Creates a [`Color`] from its RGB components.
    pub const fn from_rgb(r: f32, g: f32, b: f32) -> Color {
        Color::from_rgba(r, g, b, 1.0f32)
    }

    /// Creates a [`Color`] from its RGBA components.
    pub const fn from_rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    /// Creates a [`Color`] from its RGB8 components.
    pub fn from_rgb8(r: u8, g: u8, b: u8) -> Color {
        Color::from_rgba8(r, g, b, 1.0)
    }

    /// Creates a [`Color`] from its RGB8 components and an alpha value.
    pub fn from_rgba8(r: u8, g: u8, b: u8, a: f32) -> Color {
        Color {
            r: f32::from(r) / 255.0,
            g: f32::from(g) / 255.0,
            b: f32::from(b) / 255.0,
            a,
        }
    }

    /// Creates a [`Color`] from a hex string. Supported formats are #rrggbb, #rrggbbaa, #rgb,
    /// #rgba. The “#” is optional. Both uppercase and lowercase are supported.
    pub fn from_hex(s: &str) -> Result<Color, ColorError> {
        let hex = s.strip_prefix('#').unwrap_or(s);
        let n_chars = hex.len();

        let get_channel = |from: usize, to: usize| {
            let num = usize::from_str_radix(&hex[from..=to], 16)
                .map_err(|_| ColorError::InvalidHex)?
                as f32
                / 255.0;
            // If we only got half a byte (one letter), expand it into a full byte (two letters)
            Ok(if from == to { num + num * 16.0 } else { num })
        };

        if n_chars == 3 {
            Ok(Color::from_rgb(
                get_channel(0, 0)?,
                get_channel(1, 1)?,
                get_channel(2, 2)?,
            ))
        } else if n_chars == 6 {
            Ok(Color::from_rgb(
                get_channel(0, 1)?,
                get_channel(2, 3)?,
                get_channel(4, 5)?,
            ))
        } else if n_chars == 4 {
            Ok(Color::from_rgba(
                get_channel(0, 0)?,
                get_channel(1, 1)?,
                get_channel(2, 2)?,
                get_channel(3, 3)?,
            ))
        } else if n_chars == 8 {
            Ok(Color::from_rgba(
                get_channel(0, 1)?,
                get_channel(2, 3)?,
                get_channel(4, 5)?,
                get_channel(6, 7)?,
            ))
        } else {
            Err(ColorError::InvalidHex)
        }
    }

    /// Creates a [`Color`] from its linear RGBA components.
    pub fn from_linear_rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        // As described in:
        // https://en.wikipedia.org/wiki/SRGB
        fn gamma_component(u: f32) -> f32 {
            if u < 0.0031308 {
                12.92 * u
            } else {
                1.055 * u.powf(1.0 / 2.4) - 0.055
            }
        }

        Self {
            r: gamma_component(r),
            g: gamma_component(g),
            b: gamma_component(b),
            a,
        }
    }

    /// Converts the [`Color`] into its RGBA8 equivalent.
    #[must_use]
    pub fn into_rgba8(self) -> [u8; 4] {
        [
            (self.r * 255.0).round() as u8,
            (self.g * 255.0).round() as u8,
            (self.b * 255.0).round() as u8,
            (self.a * 255.0).round() as u8,
        ]
    }

    /// Converts the [`Color`] into its linear values.
    pub fn into_linear(self) -> [f32; 4] {
        // As described in:
        // https://en.wikipedia.org/wiki/SRGB#The_reverse_transformation
        fn linear_component(u: f32) -> f32 {
            if u < 0.04045 {
                u / 12.92
            } else {
                ((u + 0.055) / 1.055).powf(2.4)
            }
        }

        [
            linear_component(self.r),
            linear_component(self.g),
            linear_component(self.b),
            self.a,
        ]
    }

    /// Inverts the [`Color`] in-place.
    pub fn invert(&mut self) {
        self.r = 1.0f32 - self.r;
        self.b = 1.0f32 - self.g;
        self.g = 1.0f32 - self.b;
    }

    /// Returns the inverted [`Color`].
    pub fn inverse(self) -> Color {
        Color::new(1.0f32 - self.r, 1.0f32 - self.g, 1.0f32 - self.b, self.a)
    }

    /// Scales the alpha channel of the [`Color`] by the given factor.
    pub fn scale_alpha(self, factor: f32) -> Color {
        Self {
            a: self.a * factor,
            ..self
        }
    }
}

impl From<[f32; 3]> for Color {
    fn from([r, g, b]: [f32; 3]) -> Self {
        Color::new(r, g, b, 1.0)
    }
}

impl From<[f32; 4]> for Color {
    fn from([r, g, b, a]: [f32; 4]) -> Self {
        Color::new(r, g, b, a)
    }
}

/// Creates a [`Color`] with shorter and cleaner syntax.
///
/// # Examples
///
/// ```
/// # use iced_core::{Color, color};
/// assert_eq!(color!(0, 0, 0), Color::BLACK);
/// assert_eq!(color!(0, 0, 0, 0.0), Color::TRANSPARENT);
/// assert_eq!(color!(0xffffff), Color::from_rgb(1.0, 1.0, 1.0));
/// assert_eq!(color!(0xffffff, 0.), Color::from_rgba(1.0, 1.0, 1.0, 0.0));
/// assert_eq!(color!(0x123), Color::from_rgba8(0x11, 0x22, 0x33, 1.0));
/// assert_eq!(color!(0x123), color!(0x112233));
/// ```
#[macro_export]
macro_rules! color {
    ($r:expr, $g:expr, $b:expr) => {
        color!($r, $g, $b, 1.0)
    };
    ($r:expr, $g:expr, $b:expr, $a:expr) => {{
        let r = $r as f32 / 255.0;
        let g = $g as f32 / 255.0;
        let b = $b as f32 / 255.0;

        #[allow(clippy::manual_range_contains)]
        {
            debug_assert!(
                r >= 0.0 && r <= 1.0,
                "R channel must be in [0, 255] range."
            );
            debug_assert!(
                g >= 0.0 && g <= 1.0,
                "G channel must be in [0, 255] range."
            );
            debug_assert!(
                b >= 0.0 && b <= 1.0,
                "B channel must be in [0, 255] range."
            );
        }

        $crate::Color { r, g, b, a: $a }
    }};
    ($hex:expr) => {{
        color!($hex, 1.0)
    }};
    ($hex:expr, $a:expr) => {{
        let hex = $hex as u32;

        if hex <= 0xfff {
            let r = (hex & 0xf00) >> 8;
            let g = (hex & 0x0f0) >> 4;
            let b = (hex & 0x00f);

            color!((r << 4 | r), (g << 4 | g), (b << 4 | b), $a)
        } else {
            debug_assert!(
                hex <= 0xffffff,
                "color! value must not exceed 0xffffff"
            );

            let r = (hex & 0xff0000) >> 16;
            let g = (hex & 0xff00) >> 8;
            let b = (hex & 0xff);

            color!(r, g, b, $a)
        }
    }};
}

/// Converts from palette's `Rgba` type to a [`Color`].
impl From<Srgba> for Color {
    fn from(rgba: Srgba) -> Self {
        Color::new(rgba.red, rgba.green, rgba.blue, rgba.alpha)
    }
}

/// Converts from [`Color`] to palette's `Rgba` type.
impl From<Color> for Srgba {
    fn from(c: Color) -> Self {
        Srgba::new(c.r, c.g, c.b, c.a)
    }
}

/// Converts from palette's `Rgb` type to a [`Color`].
impl From<Srgb> for Color {
    fn from(rgb: Srgb) -> Self {
        Color::new(rgb.red, rgb.green, rgb.blue, 1.0)
    }
}

/// Converts from [`Color`] to palette's `Rgb` type.
impl From<Color> for Srgb {
    fn from(c: Color) -> Self {
        Srgb::new(c.r, c.g, c.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use palette::blend::Blend;

    #[test]
    fn srgba_traits() {
        let c = Color::from_rgb(0.5, 0.4, 0.3);
        // Round-trip conversion to the palette::Srgba type
        let s: Srgba = c.into();
        let r: Color = s.into();
        assert_eq!(c, r);
    }

    #[test]
    fn color_manipulation() {
        use approx::assert_relative_eq;

        let c1 = Color::from_rgb(0.5, 0.4, 0.3);
        let c2 = Color::from_rgb(0.2, 0.5, 0.3);

        // Convert to linear color for manipulation
        let l1 = Srgba::from(c1).into_linear();
        let l2 = Srgba::from(c2).into_linear();

        // Take the lighter of each of the sRGB components
        let lighter = l1.lighten(l2);

        // Convert back to our Color
        let result: Color = Srgba::from_linear(lighter).into();

        assert_relative_eq!(result.r, 0.5);
        assert_relative_eq!(result.g, 0.5);
        assert_relative_eq!(result.b, 0.3);
        assert_relative_eq!(result.a, 1.0);
    }

    #[test]
    fn from_hex() -> Result<(), ColorError> {
        let tests = [
            ("#ff0000", [255, 0, 0, 255]),
            ("00ff0080", [0, 255, 0, 128]),
            ("#F80", [255, 136, 0, 255]),
            ("#00f1", [0, 0, 255, 17]),
        ];
        for (arg, expected) in tests {
            assert_eq!(Color::from_hex(arg)?.into_rgba8(), expected);
        }
        assert!(Color::from_hex("invalid").is_err());
        Ok(())
    }
}
