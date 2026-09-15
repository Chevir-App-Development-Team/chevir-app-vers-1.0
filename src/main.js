import './styles/base.css'
import './styles/sections/navbar.css'
import './styles/sections/splash.css'
import './styles/sections/hero.css'
import './styles/sections/statement.css'
import './styles/sections/pipeline.css'
import './styles/sections/loop.css'
import './styles/sections/content.css'

import { ScrollTrigger } from 'gsap/ScrollTrigger'
import './lib/gsapSetup.js'
import { createLenis } from './lib/lenis.js'
import { splashMarkup, initPreloader } from './sections/preloader.js'
import { navbarMarkup, heroMarkup, setupHero, initNavbar } from './sections/hero.js'
import { statementMarkup, initStatement } from './sections/statement.js'
import { pipelineMarkup, initPipeline } from './sections/pipeline.js'
import { loopMarkup, initLoop } from './sections/loop.js'
import { contentMarkup, initContent } from './sections/content.js'

document.querySelector('#app').innerHTML = `
${navbarMarkup}
${splashMarkup}
${heroMarkup}
${statementMarkup}
${pipelineMarkup}
${loopMarkup}
${contentMarkup}
`

const lenis = createLenis()
const { logo } = setupHero()
initPreloader(lenis, logo, () => {})
initNavbar()
initStatement()
initPipeline()
initLoop()
initContent()

// New sections below the loop diagram add significant height and can shift
// where the pipeline/loop pins should engage; recalc after everything is laid out.
window.addEventListener('load', () => ScrollTrigger.refresh())
