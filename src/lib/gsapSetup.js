import gsap from 'gsap'
import { SplitText } from 'gsap/SplitText'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(SplitText, ScrollTrigger)

export const prefersReducedMotion = window.matchMedia(
  '(prefers-reduced-motion: reduce)'
).matches

// Pipeline və loop bölmələri eyni breakpoint dəstini bölüşür.
export const BREAKPOINTS = {
  isDesktop: '(min-width: 980px)',
  isMobile: '(max-width: 979px)',
  reduceMotion: '(prefers-reduced-motion: reduce)',
}
