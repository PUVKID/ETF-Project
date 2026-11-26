//! 프로젝트 고유 팔레트를 보존하는 로컬 UI 테마 컴파일러입니다.
use std::{collections::BTreeMap, env, process};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Color(u8, u8, u8);
impl Color {
    fn parse(s: &str) -> Result<Self, String> {
        let b = s.as_bytes();
        if b.len() != 7 || b[0] != b'#' || !b[1..].iter().all(u8::is_ascii_hexdigit) {
            return Err(format!("색상은 #RRGGBB 형식이어야 합니다: {s}"));
        }
        Ok(Self(u8::from_str_radix(&s[1..3], 16).unwrap(),
                u8::from_str_radix(&s[3..5], 16).unwrap(),
                u8::from_str_radix(&s[5..7], 16).unwrap()))
    }
    fn hex(self) -> String { format!("#{:02X}{:02X}{:02X}", self.0, self.1, self.2) }
    fn luminance(self) -> f64 {
        let f = |v: u8| { let s = v as f64 / 255.0; if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) } };
        0.2126 * f(self.0) + 0.7152 * f(self.1) + 0.0722 * f(self.2)
    }
    fn on(self) -> Self {
        // WCAG 상대 명도에 따라 포인트 배경 위 글자색을 계산합니다.
        if (self.luminance() + 0.05) / 0.05 >= 1.05 / (self.luminance() + 0.05) { Self(0, 0, 0) } else { Self(255, 255, 255) }
    }
    fn blend(self, other: Self, amount: u16) -> Self {
        let f = |a: u8, b: u8| ((a as u16 * (100 - amount) + b as u16 * amount + 50) / 100) as u8;
        Self(f(self.0, other.0), f(self.1, other.1), f(self.2, other.2))
    }
}

#[derive(Debug)]
struct Theme { format: String, mode: String, colors: BTreeMap<String, Color> }
impl Theme {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        if args.is_empty() || args.len() % 2 != 0 { return Err("--format css|qss|json|android --mode light|dark 및 색상 인수를 확인하세요".into()); }
        let mut values = BTreeMap::new();
        for pair in args.chunks_exact(2) {
            let key = pair[0].strip_prefix("--").ok_or("인수 이름은 --로 시작해야 합니다")?;
            if !["format", "mode", "accent", "background", "surface", "text", "muted", "border"].contains(&key) { return Err(format!("지원하지 않는 인수: {key}")); }
            if values.insert(key.to_string(), pair[1].clone()).is_some() { return Err(format!("중복 인수: {key}")); }
        }
        let format = values.remove("format").ok_or("--format이 필요합니다")?;
        let mode = values.remove("mode").ok_or("--mode가 필요합니다")?;
        if !["css", "qss", "json", "android"].contains(&format.as_str()) { return Err("지원하지 않는 출력 형식입니다".into()); }
        let defaults = match mode.as_str() {
            "light" => ["#4168D6", "#F2F5FA", "#FFFFFF", "#173047", "#708297", "#DCE5EF"],
            "dark" => ["#93B3FF", "#111923", "#1B2735", "#E1EAF4", "#B2C2D3", "#3C4E63"],
            _ => return Err("모드는 light 또는 dark여야 합니다".into()),
        };
        let mut colors = BTreeMap::new();
        for (name, default) in ["accent", "background", "surface", "text", "muted", "border"].iter().zip(defaults) {
            colors.insert(name.to_string(), Color::parse(values.get(*name).map(String::as_str).unwrap_or(default))?);
        }
        Ok(Self { format, mode, colors })
    }
    fn render(&self) -> String {
        let c = |k: &str| self.colors[k].hex();
        let accent = c("accent"); let bg = c("background"); let surface = c("surface");
        let text = c("text"); let muted = c("muted"); let border = c("border");
        let on = self.colors["accent"].on().hex();
        let selected = self.colors["surface"].blend(self.colors["accent"], 14).hex();
        let hover = self.colors["surface"].blend(self.colors["accent"], 7).hex();
        let disabled = self.colors["surface"].blend(self.colors["text"], 40).hex();
        match self.format.as_str() {
            "css" => {
                let selector = if self.mode == "light" { ":root,[data-theme=\"light\"],[data-mode=\"light\"]" } else { "[data-theme=\"dark\"],[data-mode=\"dark\"]" };
                format!(r#"/* Rust에서 생성한 Astryx 프로젝트 테마: {mode} */
{selector} {{
  color-scheme: {mode};
  --color-background-body: {bg}; --color-background-surface: {surface};
  --color-background-card: {surface}; --color-text-primary: {text};
  --color-text-secondary: {muted}; --color-border: {border};
  --color-border-emphasized: {accent}; --color-primary: {accent};
  --color-background-primary: {accent}; --color-text-on-primary: {on};
  --color-background-hover: {hover}; --color-background-selected: {selected};
  --color-text-disabled: {disabled};
  --font-family-body: 'NanumGothic', 'Nanum Gothic', 'Malgun Gothic', sans-serif;
  --font-family-heading: var(--font-family-body); --font-weight-body: 700;
  --spacing-1: 4px; --spacing-2: 8px; --spacing-3: 12px; --spacing-4: 16px;
  --spacing-5: 20px; --spacing-6: 24px; --spacing-8: 32px;
  --radius-inner: 6px; --radius-element: 10px; --radius-container: 16px;
  --astryx-control-height: 44px; --astryx-focus-width: 2px;
}}
"#, mode=self.mode)
            },
            "qss" => format!(r#"/* Rust에서 생성한 Astryx Qt 테마: {mode} */
QWidget {{ font-family: 'NanumGothic', 'Nanum Gothic', 'Malgun Gothic'; font-size: 14px; font-weight: 700; color: {text}; }}
QMainWindow, QDialog {{ background-color: {bg}; }}
QMenu, QToolTip {{ background-color: {surface}; color: {text}; border: 1px solid {border}; padding: 8px; }}
QPushButton, QToolButton {{ background-color: {surface}; color: {text}; border: 1px solid {border}; border-radius: 10px; min-height: 42px; padding: 0 16px; }}
QPushButton:hover, QToolButton:hover {{ background-color: {hover}; border-color: {accent}; }}
QPushButton:pressed, QToolButton:pressed {{ background-color: {selected}; }}
QPushButton:focus, QToolButton:focus, QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{ border: 2px solid {accent}; }}
QPushButton:checked, QToolButton:checked {{ background-color: {selected}; border-color: {accent}; }}
QPushButton:disabled, QToolButton:disabled, QLineEdit:disabled, QComboBox:disabled {{ color: {disabled}; background-color: {bg}; border-color: {border}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ color: {text}; background-color: {surface}; border: 1px solid {border}; border-radius: 10px; min-height: 42px; padding: 0 12px; selection-background-color: {accent}; selection-color: {on}; }}
QTextEdit, QPlainTextEdit, QListView, QTreeView, QTableView {{ background-color: {surface}; color: {text}; border: 1px solid {border}; border-radius: 10px; selection-background-color: {selected}; selection-color: {text}; }}
QAbstractItemView::item {{ min-height: 36px; padding: 4px 8px; }}
QAbstractItemView::item:hover {{ background-color: {hover}; }}
QAbstractItemView::item:selected {{ background-color: {selected}; color: {text}; }}
QHeaderView::section {{ background-color: {bg}; color: {muted}; padding: 12px; border: none; border-bottom: 1px solid {border}; }}
QGroupBox {{ background-color: {surface}; border: 1px solid {border}; border-radius: 16px; margin-top: 16px; padding: 20px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 16px; padding: 0 4px; }}
QTabBar::tab {{ min-height: 42px; padding: 0 16px; color: {muted}; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {accent}; border-bottom-color: {accent}; }}
QTabBar::tab:hover {{ background-color: {hover}; }}
QComboBox QAbstractItemView {{ background-color: {surface}; color: {text}; selection-background-color: {selected}; border: 1px solid {border}; }}
QMenu::item {{ padding: 10px 20px; border-radius: 6px; }}
QMenu::item:selected {{ background-color: {selected}; }}
QProgressBar {{ border: 1px solid {border}; border-radius: 6px; background-color: {bg}; text-align: center; min-height: 12px; }}
QProgressBar::chunk {{ background-color: {accent}; border-radius: 5px; }}
QCheckBox, QRadioButton {{ spacing: 8px; min-height: 36px; }}
QStatusBar {{ color: {muted}; background-color: {bg}; }}
QScrollBar:vertical {{ background-color: {bg}; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background-color: {border}; min-height: 32px; border-radius: 6px; }}
QScrollBar::handle:vertical:hover {{ background-color: {muted}; }}
"#, mode=self.mode),
            "android" => format!("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<resources>\n{}\n</resources>\n", self.colors.iter().map(|(k,v)| format!("  <color name=\"astryx_{k}\">{}</color>", v.hex())).collect::<Vec<_>>().join("\n")),
            _ => format!("{{\"mode\":\"{}\",\"colors\":{{{}}},\"control_height\":44,\"radius_element\":10,\"radius_container\":16,\"spacing_unit\":4,\"font_weight\":700}}\n", self.mode, self.colors.iter().map(|(k,v)| format!("\"{k}\":\"{}\"",v.hex())).collect::<Vec<_>>().join(",")),
        }
    }
}
fn main() {
    match Theme::parse(env::args().skip(1).collect()) {
        Ok(theme) => print!("{}", theme.render()),
        Err(error) => { eprintln!("테마 생성 실패: {error}"); process::exit(2); }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn args(s: &[&str]) -> Vec<String> { s.iter().map(|s| s.to_string()).collect() }
    #[test] fn invalid_colors_are_rejected() { for v in ["red", "#FFF", "#FFFFFF;", "#१२३४५६", "#GGGGGG"] { assert!(Color::parse(v).is_err()); } }
    #[test] fn input_is_preserved() { let t=Theme::parse(args(&["--format","css","--mode","dark","--accent","#00ff44"])).unwrap(); assert!(t.render().contains("--color-primary: #00FF44")); }
    #[test] fn invalid_arguments_fail() { for v in [vec![], args(&["--format","css"]), args(&["--format","css","--mode","sepia"]), args(&["--format","css","--mode","dark","--mode","light"]), args(&["--format","css","--mode","dark","--inject","x"])] { assert!(Theme::parse(v).is_err()); } }
    #[test] fn both_themes_and_formats_render() { for mode in ["light","dark"] { for format in ["css","qss","json","android"] { let s=Theme::parse(args(&["--format",format,"--mode",mode])).unwrap().render(); assert!(!s.contains("NaN")); assert!(!s.contains('\u{fffd}')); assert!(s.len()>150); } } }
    #[test] fn accent_text_contrast() { assert_eq!(Color(0,0,0).on(),Color(255,255,255)); assert_eq!(Color(255,255,255).on(),Color(0,0,0)); }
    #[test] fn blend_does_not_overflow() { assert_eq!(Color(255,255,255).blend(Color(255,255,255),40),Color(255,255,255)); }
}
