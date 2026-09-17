async function run() {
    try {
        const res = await fetch('https://www.jestdili.az/az/lugat/word/salam', {
            headers: { 'RSC': '1' }
        });
        const text = await res.text();
        const matches = text.match(/\"([^\"]*mp4[^\"]*)\"/g);
        if (matches) {
            console.log(new Set(matches));
        } else {
            console.log("No mp4 found");
            // print a bit of text to see what it is
            console.log(text.substring(0, 500));
        }
    } catch (e) {
        console.error(e);
    }
}
run();
