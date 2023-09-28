#!/usr/bin/env node

const { resolve } = require('path');
const { createReadStream } = require('fs');
const { readdir } = require('fs').promises;
const { createInterface } = require('readline');

async function getFiles(dir) {
    const dirents = await readdir(dir, { withFileTypes: true });
    const files = await Promise.all(dirents.map((dirent) => {
        const res = resolve(dir, dirent.name);
        return dirent.isDirectory() ? getFiles(res) : res;
    }));
    return files.flat();
}

console.log('module,number of nodes,tasks per node,iteration,mode,model,seed,person count,item count,iterations,burnin,batch size,mean,variance,unif,size,time in seconds,a correlation,g correlation,th correlation');

getFiles('output').then(files => {

    files.filter(f => f.endsWith('.out')).forEach(file => {

        const path = file.split('\\');
        const name = path[path.length - 1];
        const module = path[path.length - 3];
        const parts = name.split('.');

        const mpi = parts.length === 5;

        let row;

        createInterface({
            input: createReadStream(file),
            console: false
        }).on('line', function (line) {
            
            if (
                line.startsWith('run') ||
                line.startsWith('mode') ||
                line.startsWith('model') ||
                line.startsWith('seed') ||
                line.startsWith('person count') ||
                line.startsWith('item count') ||
                line.startsWith('iterations') ||
                line.startsWith('burnin') ||
                line.startsWith('batch size') ||
                line.startsWith('mean') ||
                line.startsWith('variance') ||
                line.startsWith('unif') ||
                line.startsWith('size') ||
                line.startsWith('time') ||
                line.startsWith('a') ||
                line.startsWith('g') ||
                line.startsWith('th')
            ) {
                if (line.startsWith('run')) {
                    row = mpi ? `${module},${parts[2]},${parts[3]}` : `${module},1,1`;
                }

                if (line.startsWith('time')) {
                    if (!mpi) {
                        row += `,1`;
                    }

                    row += `,${line.split(':')[1].replace('seconds', '').trim()}`
                } else {
                    row += `,${line.split(':')[1].trim()}`
                }
                if (line.startsWith('th')) {
                    console.log(row);
                }
            }

        });

    });

}).catch(e => console.error(e));
